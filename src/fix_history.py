
import logging
import sys
import pandas as pd
from datetime import datetime
from src.database import SupabaseDB
from src.ingestion import ingest_fixtures
from src.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_repair_arsenal():
    db = SupabaseDB()
    team_name = "Arsenal"
    current_season = 2024
    
    logger.info(f"Checking history for {team_name} in season {current_season}...")
    
    # query matches
    # Since we store season as integer or string (usually integer in ingestion clean_fixture_data),
    # Let's count matches where home_team=Arsenal or away_team=Arsenal AND status in ('FT', 'AET', 'PEN').
    # Also season=2024 (which is 2024-2025).
    
    try:
        # Note: We need to filter by season. In clean_fixture_data: "season": league.get("season") -> int.
        # But wait, clean_fixture_data converts it to int.
        # However, supabase filter might need checking type.
        
        # Or we can just count recent matches relative to now.
        # But user specified "2024".
        
        # Let's try select count. Postgrest doesn't support count easily without head=True.
        # We can just fetch minimal fields.
        
        response = db.client.table('matches').select('match_id').or_(
            f'home_team.eq.{team_name},away_team.eq.{team_name}'
        ).eq('season', current_season).in_('status', ['FT', 'AET', 'PEN']).execute()
        
        match_count = len(response.data)
        logger.info(f"Found {match_count} finished matches for {team_name} in season {current_season}.")
        
        if match_count < 10:
            logger.warning(f"⚠️ Insufficient history ({match_count} < 10). Triggering repair download...")
            
            # Download 2023 and 2024 (and 2025 effectively if started)
            # Premier League ID = 39
            league_id = 39 
            seasons_to_fetch = [2023, 2024]
            # API-Football 2025 season might not exist for PL yet if currently Feb 2026? 
            # Current date is 2026-02-18. So 2025-2026 season matches should be fetched.
            # user said "2023, 2024 et 2025".
            seasons_to_fetch.append(2025)
            
            for season in seasons_to_fetch:
                logger.info(f"⬇️ Downloading Premier League season {season}...")
                try:
                    ingest_fixtures(league_id, season, fetch_stats=True)
                    logger.info(f"✅ Downloaded season {season}.")
                except Exception as e:
                    logger.error(f"❌ Failed to download season {season}: {e}")
            
            logger.info("Repair complete. Validating stats...")
            validate_stats(team_name)
            
        else:
            logger.info(f"✅ History looks sufficient ({match_count} matches). No repair needed.")
            # Still validate to be sure
            validate_stats(team_name)

    except Exception as e:
        logger.error(f"Error Checking/Repairing history: {e}")

def validate_stats(team_name):
    """
    Recalculate features in-memory and verify Attack Strength > 1.0.
    """
    logger.info("Recalculating rolling features to verify fix...")
    fe = FeatureEngineer()
    
    # Load data (enough to form rolling stats)
    matches, stats = fe.load_data(limit=1000, completed_only=True)
    
    if matches.empty:
        logger.error("No data loaded for validation.")
        return
        
    # Process
    df = fe.process_all_features(matches, stats)
    
    # Filter for team
    team_rows = df[df['team'] == team_name].sort_values('match_date')
    
    if team_rows.empty:
        logger.error(f"No rows found for {team_name} in processed data.")
        return
        
    last_row = team_rows.iloc[-1]
    att_strength = last_row.get('attack_strength', 0.0)
    match_date = last_row['match_date']
    opponent = last_row['opponent']
    
    logger.info(f"Latest Match: {match_date} vs {opponent}")
    logger.info(f"Attack Strength: {att_strength:.3f}")
    
    if att_strength > 1.05:
        logger.info("✅ SUCCESS: Attack Strength is realistic (> 1.0).")
    elif att_strength == 1.0:
        logger.warning("❌ WARNING: Attack Strength is exactly 1.0 (Default). History might still be disconnected or insufficient.")
    else:
        logger.info(f"ℹ️ Note: Attack Strength is {att_strength:.3f} (Lower than average but not default).")

if __name__ == "__main__":
    check_and_repair_arsenal()
