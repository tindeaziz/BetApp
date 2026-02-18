"""
Database module for Supabase connection and table management.
Implements Singleton pattern for connection pooling.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from postgrest import SyncPostgrestClient
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SupabaseDB:
    """
    Singleton class for managing Supabase database connection.
    Ensures only one connection instance exists throughout the application.
    """
    
    _instance: Optional['SupabaseDB'] = None
    _client: Optional[SyncPostgrestClient] = None
    
    def __new__(cls) -> 'SupabaseDB':
        """Create or return existing instance (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Supabase client if not already initialized."""
        if self._client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
                )
            
            # Construct the PostgREST URL (typically URL + '/rest/v1')
            postgrest_url = f"{url}/rest/v1"
            
            self._client = SyncPostgrestClient(
                base_url=postgrest_url,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}"
                }
            )
            logger.info("Supabase client initialized successfully")
    
    @property
    def client(self) -> SyncPostgrestClient:
        """Get the Supabase client instance."""
        if self._client is None:
            raise RuntimeError("Database client not initialized")
        return self._client


def init_db() -> None:
    """
    Initialize database schema by creating necessary tables if they don't exist.
    Creates three main tables: matches, stats, and predictions.
    
    Raises:
        Exception: If table creation fails
    """
    db = SupabaseDB()
    
    # SQL for creating matches table
    matches_table_sql = """
    CREATE TABLE IF NOT EXISTS matches (
        id SERIAL PRIMARY KEY,
        match_id INTEGER UNIQUE NOT NULL,
        home_team VARCHAR(100) NOT NULL,
        away_team VARCHAR(100) NOT NULL,
        home_logo VARCHAR(255),
        away_logo VARCHAR(255),
        home_score INTEGER,
        away_score INTEGER,
        match_date TIMESTAMP NOT NULL,
        league VARCHAR(100),
        season VARCHAR(20),
        referee VARCHAR(100),
        venue VARCHAR(100),
        result VARCHAR(10),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_matches_match_id ON matches(match_id);
    CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
    """
    
    # SQL for creating stats table
    stats_table_sql = """
    CREATE TABLE IF NOT EXISTS stats (
        id SERIAL PRIMARY KEY,
        match_id INTEGER NOT NULL,
        team VARCHAR(100) NOT NULL,
        is_home BOOLEAN NOT NULL,
        shots_on_target INTEGER DEFAULT 0,
        shots_off_target INTEGER DEFAULT 0,
        total_shots INTEGER DEFAULT 0,
        blocked_shots INTEGER DEFAULT 0,
        shots_inside_box INTEGER DEFAULT 0,
        shots_outside_box INTEGER DEFAULT 0,
        fouls INTEGER DEFAULT 0,
        corner_kicks INTEGER DEFAULT 0,
        offsides INTEGER DEFAULT 0,
        ball_possession INTEGER DEFAULT 0,
        yellow_cards INTEGER DEFAULT 0,
        red_cards INTEGER DEFAULT 0,
        goalkeeper_saves INTEGER DEFAULT 0,
        total_passes INTEGER DEFAULT 0,
        passes_accurate INTEGER DEFAULT 0,
        passes_percentage INTEGER DEFAULT 0,
        expected_goals FLOAT DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    );
    
    CREATE INDEX IF NOT EXISTS idx_stats_match_id ON stats(match_id);
    CREATE INDEX IF NOT EXISTS idx_stats_team ON stats(team);
    """
    
    # SQL for creating predictions table
    predictions_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        match_id INTEGER,
        home_team VARCHAR(100) NOT NULL,
        away_team VARCHAR(100) NOT NULL,
        prediction_date TIMESTAMP DEFAULT NOW(),
        prob_home_win FLOAT,
        prob_draw FLOAT,
        prob_away_win FLOAT,
        predicted_result VARCHAR(10),
        predicted_over_under FLOAT,
        predicted_fouls INTEGER,
        confidence_score FLOAT,
        model_version VARCHAR(50),
        actual_result VARCHAR(10),
        is_correct BOOLEAN,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
    """
    
    try:
        # Execute SQL via Supabase RPC or direct SQL execution
        # Note: Supabase Python client doesn't have direct SQL execution
        # You'll need to run these via Supabase Dashboard SQL Editor or use psycopg2
        logger.info("Database tables schema defined. Please execute the following SQL in Supabase Dashboard:")
        logger.info("\n" + matches_table_sql)
        logger.info("\n" + stats_table_sql)
        logger.info("\n" + predictions_table_sql)
        
        # Alternative: Check if tables exist by trying to query them
        try:
            db.client.table('matches').select('id').limit(1).execute()
            logger.info("Table 'matches' exists")
        except Exception:
            logger.warning("Table 'matches' may not exist. Please create it using the SQL above.")
        
        try:
            db.client.table('stats').select('id').limit(1).execute()
            logger.info("Table 'stats' exists")
        except Exception:
            logger.warning("Table 'stats' may not exist. Please create it using the SQL above.")
        
        try:
            db.client.table('predictions').select('id').limit(1).execute()
            logger.info("Table 'predictions' exists")
        except Exception:
            logger.warning("Table 'predictions' may not exist. Please create it using the SQL above.")
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


def upsert_match(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update match data in the database.
    
    Args:
        match_data: Dictionary containing match information
        
    Returns:
        Response from database operation
        
    Raises:
        Exception: If upsert operation fails
    """
    db = SupabaseDB()
    
    try:
        response = db.client.table('matches').upsert(
            match_data,
            on_conflict='match_id'
        ).execute()
        logger.info(f"Match {match_data.get('match_id')} upserted successfully")
        return response.data
    except Exception as e:
        logger.error(f"Error upserting match: {e}")
        raise


def upsert_stats(stats_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Insert or update match statistics in the database.
    
    Args:
        stats_data: List of dictionaries containing match statistics
        
    Returns:
        Response from database operation
        
    Raises:
        Exception: If upsert operation fails
    """
    db = SupabaseDB()
    
    try:
        response = db.client.table('stats').upsert(stats_data).execute()
        logger.info(f"Stats for {len(stats_data)} teams upserted successfully")
        return response.data
    except Exception as e:
        logger.error(f"Error upserting stats: {e}")
        raise


def insert_prediction(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert prediction data into the database.
    
    Args:
        prediction_data: Dictionary containing prediction information
        
    Returns:
        Response from database operation
        
    Raises:
        Exception: If insert operation fails
    """
    db = SupabaseDB()
    
    try:
        response = db.client.table('predictions').insert(prediction_data).execute()
        logger.info(f"Prediction for match {prediction_data.get('match_id')} inserted")
        return response.data
    except Exception as e:
        logger.error(f"Error inserting prediction: {e}")
        raise


def get_team_matches(team_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recent matches for a specific team.
    
    Args:
        team_name: Name of the team
        limit: Maximum number of matches to retrieve
        
    Returns:
        List of match records
        
    Raises:
        Exception: If query fails
    """
    db = SupabaseDB()
    
    try:
        response = db.client.table('matches').select('*').or_(
            f'home_team.eq.{team_name},away_team.eq.{team_name}'
        ).order('match_date', desc=True).limit(limit).execute()
        
        return response.data
    except Exception as e:
        logger.error(f"Error retrieving matches for team {team_name}: {e}")
        raise


def get_team_stats(team_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recent statistics for a specific team.
    
    Args:
        team_name: Name of the team
        limit: Maximum number of stat records to retrieve
        
    Returns:
        List of statistics records
        
    Raises:
        Exception: If query fails
    """
    db = SupabaseDB()
    
    try:
        response = db.client.table('stats').select('*').eq(
            'team', team_name
        ).order('created_at', desc=True).limit(limit).execute()
        
        return response.data
    except Exception as e:
        logger.error(f"Error retrieving stats for team {team_name}: {e}")
        raise


def _competition_icon(league_name: str) -> str:
    """Return an emoji icon for a given competition name."""
    icons = {
        'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'Ligue 1': '🇫🇷',
        'La Liga': '🇪🇸',
        'Bundesliga': '🇩🇪',
        'Serie A': '🇮🇹',
        'Eredivisie': '🇳🇱',
        'Primeira Liga': '🇵🇹',
        'Champions League': '🏆',
        'UEFA Champions League': '🏆',
        'Europa League': '🥈',
        'UEFA Europa League': '🥈',
        'Conference League': '🥉',
    }
    for key, icon in icons.items():
        if key.lower() in league_name.lower():
            return icon
    return '⚽'


def get_todays_matches() -> List[Dict[str, Any]]:
    """
    Retrieve matches scheduled for TODAY only.
    Queries for matches where match_date falls on today's date
    and result is NULL (not yet played).
    
    Returns:
        List of dicts with: id, label, home, away, match_date, league, venue, referee
    """
    from datetime import datetime

    db = SupabaseDB()

    try:
        today_start = datetime.now().strftime("%Y-%m-%dT00:00:00")
        today_end = datetime.now().strftime("%Y-%m-%dT23:59:59")

        response = db.client.table('matches').select('*').is_(
            'result', 'null'
        ).gte(
            'match_date', today_start
        ).lte(
            'match_date', today_end
        ).order('match_date', desc=False).execute()

        matches = response.data
        todays = []

        for match in matches:
            match_date = match.get('match_date', '')
            league_name = match.get('league', '')
            try:
                dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                # If time is exactly 00:00, it's often a placeholder for TBD in some feeds
                if dt.hour == 0 and dt.minute == 0:
                     date_label = "TBD"
                else:
                     date_label = dt.strftime("%H:%M")
            except Exception:
                date_label = "TBD"

            # Competition icon
            comp_icon = _competition_icon(league_name)

            todays.append({
                'id': match.get('match_id'),
                'label': f"{comp_icon} {date_label} — {match.get('home_team')} vs {match.get('away_team')}",
                'time': date_label,  # Added for app.py
                'home': match.get('home_team'),
                'away': match.get('away_team'),
                'logo_home': match.get('home_logo'), # Renamed for app.py
                'logo_away': match.get('away_logo'), # Renamed for app.py
                'match_date': match_date,
                'league': league_name,
                'league_id': match.get('league_id'),
                'venue': match.get('venue', ''),
                'referee': match.get('referee'),
                'odds_home': match.get('odds_home'),
                'odds_draw': match.get('odds_draw'),
                'odds_away': match.get('odds_away'),
            })

        logger.info(f"Found {len(todays)} matches today ({datetime.now().date()})")
        return todays

    except Exception as e:
        logger.error(f"Error retrieving today's matches: {e}")
        return []


def get_cached_prediction(match_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached prediction for a given match from Supabase.
    If a prediction already exists for this match_id, return it
    so we don't recompute for every user.

    Args:
        match_id: The fixture/match ID

    Returns:
        Prediction dict if cached, None otherwise
    """
    db = SupabaseDB()

    try:
        response = db.client.table('predictions').select('*').eq(
            'match_id', match_id
        ).limit(1).execute()

        if response.data:
            row = response.data[0]
            logger.info(f"Cache HIT for match {match_id}")
            return {
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'prob_home': float(row.get('prob_home_win', 0)),
                'prob_draw': float(row.get('prob_draw', 0)),
                'prob_away': float(row.get('prob_away_win', 0)),
                'predicted_result': row.get('predicted_result'),
                'confidence': float(row.get('confidence_score', 0)),
                'predicted_goals': float(row.get('predicted_over_under', 0)),
                'predicted_fouls': int(row.get('predicted_fouls', 0)),
                'cached': True
            }

        logger.info(f"Cache MISS for match {match_id}")
        return None

    except Exception as e:
        logger.warning(f"Error checking prediction cache for match {match_id}: {e}")
        return None


def save_prediction_cache(match_id: int, prediction: Dict[str, Any]) -> None:
    """
    Save a prediction to Supabase so other users can reuse it.

    Args:
        match_id: The fixture/match ID
        prediction: Prediction dict with probabilities, goals, fouls, etc.
    """
    from datetime import datetime

    db = SupabaseDB()

    try:
        row = {
            'match_id': match_id,
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'prob_home_win': prediction['prob_home'],
            'prob_draw': prediction['prob_draw'],
            'prob_away_win': prediction['prob_away'],
            'predicted_result': prediction['predicted_result'],
            'predicted_over_under': prediction['predicted_goals'],
            'predicted_fouls': prediction['predicted_fouls'],
            'confidence_score': prediction['confidence'],
            'model_version': 'v1.0',
            'prediction_date': datetime.now().isoformat()
        }

        db.client.table('predictions').upsert(row).execute()
        logger.info(f"Prediction cached for match {match_id}")

    except Exception as e:
        logger.warning(f"Error saving prediction cache for match {match_id}: {e}")



def ingest_upcoming_fixtures(league_id: int, season: int, next_n: int = 20) -> int:
    """
    Fetch and store upcoming fixtures from API-Football.
    
    Args:
        league_id: League ID
        season: Season year
        next_n: Number of next fixtures to fetch
        
    Returns:
        Number of fixtures ingested
    """
    from src.ingestion import APIFootballClient, clean_fixture_data
    import time
    
    client = APIFootballClient()
    ingested_count = 0
    
    try:
        raw_fixtures = client.fetch_next_fixtures(league_id, season, next_n)
        
        for raw_fixture in raw_fixtures:
            try:
                cleaned = clean_fixture_data(raw_fixture)
                
                # Fetch odds
                fixture_id = cleaned.get('match_id')
                if fixture_id:
                    odds = client.fetch_odds(fixture_id)
                    if odds:
                        cleaned['odds_home'] = odds.get('odds_home')
                        cleaned['odds_draw'] = odds.get('odds_draw')
                        cleaned['odds_away'] = odds.get('odds_away')
                        # Log if we successfully got odds
                        logger.info(f"  > Odds found: {odds}")
                    else:
                        logger.info("  > No odds available yet")
                
                upsert_match(cleaned)
                ingested_count += 1
                logger.info(f"Ingested upcoming: {cleaned['home_team']} vs {cleaned['away_team']}")
                
                # Respect API rate limits
                time.sleep(0.25)
                
            except Exception as e:
                logger.error(f"Error ingesting upcoming fixture: {e}")
                continue
        
        logger.info(f"Successfully ingested {ingested_count} upcoming fixtures")
        return ingested_count
        
    except Exception as e:
        logger.error(f"Error during upcoming ingestion: {e}")
        raise

