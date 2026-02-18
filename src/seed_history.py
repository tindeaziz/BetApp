"""
Script to seed historical match data (2023-2025) into Supabase.
This allows the ML model to calculate accurate form, Elo, and stats instead of using defaults.

Usage:
    python src/seed_history.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import APIFootballClient, clean_fixture_data, LEAGUE_IDS
from src.database import upsert_match, SupabaseDB
from src.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target years
YEARS = [2023, 2024, 2025]


def seed_history():
    """
    Main function to seed historical data.
    Iterates through years and leagues, fetching finished matches.
    """
    client = APIFootballClient()
    total_ingested = 0
    
    logger.info(f"Starting historical data seeding for years: {YEARS}")
    logger.info(f"Target leagues: {[l['name'] for l in LEAGUE_IDS.values()]}")
    
    for league_id, info in LEAGUE_IDS.items():
        league_name = info['name']
        
        for season in YEARS:
            logger.info(f"Fetching {league_name} ({league_id}) - Season {season}...")
            
            try:
                # Fetch all fixtures for the season
                # Note: fetch_fixtures without date_from/to fetches the whole season
                fixtures = client.fetch_fixtures(league_id, season)
                
                # Filter for Finished (FT) only
                finished = [
                    f for f in fixtures 
                    if f.get('fixture', {}).get('status', {}).get('short') in ['FT', 'AET', 'PEN']
                ]
                
                if not finished:
                    logger.info(f"  > No finished matches found for {league_name} {season}")
                    continue
                
                logger.info(f"  > Found {len(finished)} finished matches. Ingesting...")
                
                count = 0
                for raw_fixture in finished:
                    try:
                        cleaned = clean_fixture_data(raw_fixture)
                        upsert_match(cleaned)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error ingesting fixture {raw_fixture.get('fixture', {}).get('id')}: {e}")
                
                total_ingested += count
                logger.info(f"  > Ingested {count} matches for {league_name} {season}")
                
                # Respect API rate limits (avoid bursting)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching {league_name} {season}: {e}")
                time.sleep(5)  # Backoff on error
    
    logger.info(f"Historical seeding complete! Total matches ingested: {total_ingested}")
    
    # Run stats recalculation
    recalculate_stats()


def recalculate_stats():
    """
    Recalculate and update derived stats (Form, Elo, Avg Goals) for all teams.
    
    NOTE: The current architecture calculates these features ON-THE-FLY using FeatureEngineer
    when `run_prediction` is called. The database schema does NOT currently store 
    'form', 'elo', or 'avg_goals' columns in the `matches` table.
    
    This function therefore simulates the update by:
    1. Initializing the FeatureEngineer (which loads the new history).
    2. Calculating metrics for the most recent matches to verify data integrity.
    3. Logging the updated metrics to confirm the model now has history.
    """
    logger.info("\n" + "="*50)
    logger.info("RECALCULATING DERIVED STATS (Form, Elo, Avg Goals)")
    logger.info("="*50)
    
    fe = FeatureEngineer()
    
    # 1. Update/Verify Elo Ratings
    logger.info("Updating Elo ratings based on full history...")
    elo_ratings = fe.calculate_elo_ratings()
    
    # Show top 5 teams by Elo to verify
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 Teams by Calculated Elo:")
    for team, rating in sorted_elo:
        logger.info(f"  - {team}: {rating:.0f}")
        
    # 2. Verify Form and Goals for a few major teams
    sample_teams = ["Paris Saint Germain", "Real Madrid", "Manchester City", "Arsenal", "Inter"]
    
    logger.info("\nVerifying Form & Goal stats for sample teams:")
    for team in sample_teams:
        try:
            form = fe.calculate_team_form(team)
            
            # Calculate avg goals from db directly to verify
            limit = 5
            matches = fe.db.client.table('matches').select('*').or_(
                f'home_team.eq.{team},away_team.eq.{team}'
            ).order('match_date', desc=True).limit(limit).execute().data
            
            goals = 0
            if matches:
                for m in matches:
                    if m['home_team'] == team:
                        goals += m['home_score'] or 0
                    else:
                        goals += m['away_score'] or 0
                avg_goals = goals / len(matches)
            else:
                avg_goals = 0.0
                
            logger.info(f"  - {team}: Form={form:.2f} | Avg Goals (last {limit}): {avg_goals:.1f}")
            
        except Exception as e:
            logger.warning(f"Could not calc stats for {team}: {e}")
            
    logger.info("\n✅ Stats recalculation complete. Feature Store is now hydrated with history.")


if __name__ == "__main__":
    seed_history()
