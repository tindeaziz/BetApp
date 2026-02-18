"""
Data ingestion module for fetching and cleaning football match data.
Simulates API-Football integration with generic interface.
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import logging

from src.database import upsert_match, upsert_stats

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────
# CONFIGURABLE LEAGUE IDS
# Add any league ID here to include it in ingestion.
# ─────────────────────────────────────────────────────
LEAGUE_IDS = {
    39:  {"name": "Premier League",    "season": 2025, "type": "LEAGUE"},
    61:  {"name": "Ligue 1",           "season": 2025, "type": "LEAGUE"},
    140: {"name": "La Liga",           "season": 2025, "type": "LEAGUE"},
    78:  {"name": "Bundesliga",        "season": 2025, "type": "LEAGUE"},
    135: {"name": "Serie A",           "season": 2025, "type": "LEAGUE"},
    88:  {"name": "Eredivisie",        "season": 2025, "type": "LEAGUE"},
    94:  {"name": "Primeira Liga",     "season": 2025, "type": "LEAGUE"},
    2:   {"name": "Champions League",  "season": 2025, "type": "CUP"},
    3:   {"name": "Europa League",     "season": 2025, "type": "CUP"},
}

# League IDs considered cup competitions
CUP_LEAGUE_IDS = {lid for lid, info in LEAGUE_IDS.items() if info["type"] == "CUP"}


class APIFootballClient:
    """
    Client for interacting with API-Football.
    Provides methods to fetch matches, statistics, and other data.
    """
    
    def __init__(self):
        """Initialize API client with credentials from environment."""
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.base_url = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
        self.headers = {
            "x-apisports-key": self.api_key
        }
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.headers)
    
    def fetch_fixtures(
        self,
        league_id: int,
        season: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch fixtures (matches) from API-Football.
        
        Args:
            league_id: ID of the league
            season: Season year (e.g., 2023)
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            
        Returns:
            List of fixture data dictionaries
            
        Raises:
            Exception: If API request fails
        """
        endpoint = f"{self.base_url}/fixtures"
        params: Dict[str, Any] = {
            "league": league_id,
            "season": season
        }
        
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("errors"):
                logger.error(f"API returned errors: {data['errors']}")
                return []
            
            logger.info(f"Fetched {len(data.get('response', []))} fixtures")
            return data.get("response", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching fixtures: {e}")
            raise
    
    def fetch_next_fixtures(
        self,
        league_id: int,
        season: int,
        next_n: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch upcoming / not-started fixtures from API-Football.
        
        Args:
            league_id: ID of the league
            season: Season year
            next_n: Number of next fixtures to fetch
            
        Returns:
            List of upcoming fixture data dictionaries
        """
        endpoint = f"{self.base_url}/fixtures"
        params: Dict[str, Any] = {
            "league": league_id,
            "season": season,
            "next": next_n
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("errors"):
                logger.error(f"API returned errors: {data['errors']}")
                return []
            
            fixtures = data.get("response", [])
            logger.info(f"Fetched {len(fixtures)} upcoming fixtures")
            return fixtures
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching next fixtures: {e}")
            raise
    
    def fetch_odds(self, fixture_id: int) -> Dict[str, float]:
        """
        Fetch pre-match odds (1X2 — Match Winner) for a given fixture.
        Uses the first available bookmaker.

        Args:
            fixture_id: API-Football fixture ID

        Returns:
            Dict with keys odds_home, odds_draw, odds_away (floats).
            Returns empty dict if unavailable.
        """
        endpoint = f"{self.base_url}/odds"
        params = {"fixture": fixture_id}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("errors") or not data.get("response"):
                return {}

            # Navigate: response[0] -> bookmakers[0] -> bets -> find "Match Winner"
            bookmakers = data["response"][0].get("bookmakers", [])
            if not bookmakers:
                return {}

            for bookmaker in bookmakers:
                for bet in bookmaker.get("bets", []):
                    if bet.get("name") == "Match Winner":
                        values = bet.get("values", [])
                        odds = {}
                        for v in values:
                            val = v.get("value")
                            odd = v.get("odd")
                            if val == "Home":
                                odds["odds_home"] = float(odd)
                            elif val == "Draw":
                                odds["odds_draw"] = float(odd)
                            elif val == "Away":
                                odds["odds_away"] = float(odd)
                        if odds:
                            logger.info(f"Odds for fixture {fixture_id}: H={odds.get('odds_home')} D={odds.get('odds_draw')} A={odds.get('odds_away')}")
                            return odds

            return {}

        except Exception as e:
            logger.warning(f"Could not fetch odds for fixture {fixture_id}: {e}")
            return {}

    def fetch_injuries(self, fixture_id: int) -> Dict[str, list]:
        """
        Fetch injured/unavailable players for a fixture.

        Args:
            fixture_id: API-Football fixture ID

        Returns:
            Dict with team names as keys, lists of injured player dicts as values.
            Each player dict: {"name": str, "reason": str, "type": str}
        """
        endpoint = f"{self.base_url}/injuries"
        params = {"fixture": fixture_id}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("errors") or not data.get("response"):
                return {}

            injuries: Dict[str, list] = {}
            for entry in data["response"]:
                team_name = entry.get("team", {}).get("name", "Unknown")
                player_name = entry.get("player", {}).get("name", "Unknown")
                reason = entry.get("player", {}).get("reason", "")
                injury_type = entry.get("player", {}).get("type", "")

                if team_name not in injuries:
                    injuries[team_name] = []
                injuries[team_name].append({
                    "name": player_name,
                    "reason": reason,
                    "type": injury_type,
                })

            for team, players in injuries.items():
                logger.info(f"Injuries for {team}: {[p['name'] for p in players]}")

            return injuries

        except Exception as e:
            logger.warning(f"Could not fetch injuries for fixture {fixture_id}: {e}")
            return {}

    def fetch_top_scorers(self, league_id: int, season: int, limit: int = 5) -> Dict[str, str]:
        """
        Fetch top scorers for a league/season.

        Args:
            league_id: League ID
            season: Season year
            limit: Number of top scorers to return

        Returns:
            Dict mapping team_name -> top_scorer_name
        """
        endpoint = f"{self.base_url}/players/topscorers"
        params = {"league": league_id, "season": season}

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("errors") or not data.get("response"):
                return {}

            team_top_scorer: Dict[str, str] = {}
            for entry in data["response"][:20]:  # scan top 20 to cover more teams
                player = entry.get("player", {}).get("name", "")
                # Get the team from statistics
                stats = entry.get("statistics", [])
                if stats:
                    team = stats[0].get("team", {}).get("name", "")
                    if team and team not in team_top_scorer:
                        team_top_scorer[team] = player

            logger.info(f"Top scorers for league {league_id}: {team_top_scorer}")
            return team_top_scorer

        except Exception as e:
            logger.warning(f"Could not fetch top scorers for league {league_id}: {e}")
            return {}

    def fetch_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Fetch detailed statistics for a specific fixture.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            List of dictionaries containing match statistics
            
        Raises:
            Exception: If API request fails
        """
        endpoint = f"{self.base_url}/fixtures/statistics"
        params = {"fixture": fixture_id}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("errors"):
                logger.error(f"API returned errors: {data['errors']}")
                return []
            
            return data.get("response", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching statistics for fixture {fixture_id}: {e}")
            raise


def clean_fixture_data(raw_fixture: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and transform raw fixture data into database format.
    
    Args:
        raw_fixture: Raw fixture data from API
        
    Returns:
        Cleaned fixture data ready for database insertion
    """
    try:
        fixture = raw_fixture.get("fixture", {})
        teams = raw_fixture.get("teams", {})
        goals = raw_fixture.get("goals", {})
        league = raw_fixture.get("league", {})
        
        # Detect match status: FT=finished, NS=not started, TBD=to be defined
        status_info = fixture.get("status", {})
        status_short = status_info.get("short", "FT")
        
        # For upcoming matches, scores are null
        is_upcoming = status_short in ("NS", "TBD", "PST", "CANC")
        home_score = None if is_upcoming else goals.get("home")
        away_score = None if is_upcoming else goals.get("away")
        
        cleaned_data = {
            "match_id": fixture.get("id"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "home_score": home_score,
            "away_score": away_score,
            "home_logo": teams.get("home", {}).get("logo"),
            "away_logo": teams.get("away", {}).get("logo"),
            "match_date": fixture.get("date"),
            "league": league.get("name"),
            "league_id": league.get("id"),
            "season": league.get("season"),
            "referee": fixture.get("referee"),
            "venue": fixture.get("venue", {}).get("name"),
            "result": determine_result(home_score, away_score),
            "status": status_short,
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
        }
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error cleaning fixture data: {e}")
        raise


def clean_statistics_data(
    raw_stats: List[Dict[str, Any]],
    match_id: int
) -> List[Dict[str, Any]]:
    """
    Clean and transform raw statistics data into database format.
    
    Args:
        raw_stats: Raw statistics data from API (list with home and away team stats)
        match_id: ID of the match
        
    Returns:
        List of cleaned statistics dictionaries (one per team)
    """
    cleaned_stats = []
    
    try:
        for team_stats in raw_stats:
            team_name = team_stats.get("team", {}).get("name")
            statistics = team_stats.get("statistics", [])
            
            # Convert list of statistics to dictionary
            stats_dict = {}
            for stat in statistics:
                stat_type = stat.get("type")
                stat_value = stat.get("value")
                stats_dict[stat_type] = stat_value
            
            # Extract and normalize statistics
            cleaned = {
                "match_id": match_id,
                "team": team_name,
                "is_home": team_stats.get("team", {}).get("id") == raw_stats[0].get("team", {}).get("id"),
                "shots_on_target": safe_int(stats_dict.get("Shots on Goal", 0)),
                "shots_off_target": safe_int(stats_dict.get("Shots off Goal", 0)),
                "total_shots": safe_int(stats_dict.get("Total Shots", 0)),
                "blocked_shots": safe_int(stats_dict.get("Blocked Shots", 0)),
                "shots_inside_box": safe_int(stats_dict.get("Shots insidebox", 0)),
                "shots_outside_box": safe_int(stats_dict.get("Shots outsidebox", 0)),
                "fouls": safe_int(stats_dict.get("Fouls", 0)),
                "corner_kicks": safe_int(stats_dict.get("Corner Kicks", 0)),
                "offsides": safe_int(stats_dict.get("Offsides", 0)),
                "ball_possession": safe_int(stats_dict.get("Ball Possession", "0%").replace("%", "")),
                "yellow_cards": safe_int(stats_dict.get("Yellow Cards", 0)),
                "red_cards": safe_int(stats_dict.get("Red Cards", 0)),
                "goalkeeper_saves": safe_int(stats_dict.get("Goalkeeper Saves", 0)),
                "total_passes": safe_int(stats_dict.get("Total passes", 0)),
                "passes_accurate": safe_int(stats_dict.get("Passes accurate", 0)),
                "passes_percentage": safe_int(stats_dict.get("Passes %", "0%").replace("%", "")),
                "expected_goals": safe_float(stats_dict.get("expected_goals", 0.0))
            }
            
            cleaned_stats.append(cleaned)
        
        return cleaned_stats
        
    except Exception as e:
        logger.error(f"Error cleaning statistics data: {e}")
        raise


def determine_result(home_score: Optional[int], away_score: Optional[int]) -> Optional[str]:
    """
    Determine match result based on scores.
    
    Args:
        home_score: Home team score
        away_score: Away team score
        
    Returns:
        Result string: 'H' (home win), 'D' (draw), 'A' (away win), or None
    """
    if home_score is None or away_score is None:
        return None
    
    # Asserting not None for type checkers
    assert home_score is not None
    assert away_score is not None

    if home_score > away_score:
        return 'H'
    elif home_score < away_score:
        return 'A'
    else:
        return 'D'


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        if value is None or value == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def ingest_fixtures(
    league_id: int,
    season: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fetch_stats: bool = True
) -> int:
    """
    Main ingestion function to fetch and store fixtures and statistics.
    
    Args:
        league_id: ID of the league
        season: Season year
        date_from: Start date (optional)
        date_to: End date (optional)
        fetch_stats: Whether to fetch detailed statistics for each match
        
    Returns:
        Number of fixtures ingested
        
    Raises:
        Exception: If ingestion process fails
    """
    client = APIFootballClient()
    ingested_count = 0
    
    try:
        # Fetch fixtures
        logger.info(f"Fetching fixtures for league {league_id}, season {season}")
        fixtures = client.fetch_fixtures(league_id, season, date_from, date_to)
        
        for raw_fixture in fixtures:
            try:
                # Clean and insert match data
                cleaned_fixture = clean_fixture_data(raw_fixture)
                upsert_match(cleaned_fixture)
                
                # Only fetch stats for finished matches (not NS/TBD)
                match_status = cleaned_fixture.get("status", "FT")
                is_finished = match_status in ("FT", "AET", "PEN")
                
                if fetch_stats and is_finished:
                    match_id = cleaned_fixture["match_id"]
                    logger.info(f"Fetching statistics for match {match_id}")
                    
                    raw_stats = client.fetch_statistics(match_id)
                    if raw_stats:
                        cleaned_stats = clean_statistics_data(raw_stats, match_id)
                        upsert_stats(cleaned_stats)
                elif not is_finished:
                    logger.info(f"Skipping stats for upcoming match {cleaned_fixture['match_id']} (status: {match_status})")
                
                # Fetch odds for upcoming matches
                if not is_finished:
                    odds = client.fetch_odds(cleaned_fixture["match_id"])
                    if odds:
                        cleaned_fixture.update(odds)
                        upsert_match(cleaned_fixture)  # re-upsert with odds
                
                ingested_count += 1
                logger.info(f"Ingested match {cleaned_fixture['match_id']}: "
                          f"{cleaned_fixture['home_team']} vs {cleaned_fixture['away_team']} [{match_status}]")
                
            except Exception as e:
                logger.error(f"Error ingesting fixture: {e}")
                continue
        
        logger.info(f"Successfully ingested {ingested_count} fixtures")
        return ingested_count
        
    except Exception as e:
        logger.error(f"Error during ingestion process: {e}")
        raise


def simulate_fetch_data() -> Dict[str, Any]:
    """
    Simulate API data fetching for testing purposes.
    Returns mock data in API-Football format.
    
    Returns:
        Dictionary with simulated fixture and statistics data
    """
    simulated_fixture = {
        "fixture": {
            "id": 999999,
            "referee": "John Doe",
            "date": "2024-02-16T20:00:00+00:00",
            "venue": {
                "name": "Stadium Name"
            }
        },
        "league": {
            "name": "Premier League",
            "season": 2023
        },
        "teams": {
            "home": {"id": 1, "name": "Manchester United"},
            "away": {"id": 2, "name": "Liverpool"}
        },
        "goals": {
            "home": 2,
            "away": 1
        }
    }
    
    simulated_stats = [
        {
            "team": {"id": 1, "name": "Manchester United"},
            "statistics": [
                {"type": "Shots on Goal", "value": 6},
                {"type": "Shots off Goal", "value": 4},
                {"type": "Total Shots", "value": 12},
                {"type": "Fouls", "value": 11},
                {"type": "Corner Kicks", "value": 5},
                {"type": "Ball Possession", "value": "55%"},
                {"type": "Yellow Cards", "value": 2},
                {"type": "Total passes", "value": 450},
                {"type": "Passes accurate", "value": 380},
                {"type": "Passes %", "value": "84%"}
            ]
        },
        {
            "team": {"id": 2, "name": "Liverpool"},
            "statistics": [
                {"type": "Shots on Goal", "value": 4},
                {"type": "Shots off Goal", "value": 3},
                {"type": "Total Shots", "value": 9},
                {"type": "Fouls", "value": 13},
                {"type": "Corner Kicks", "value": 3},
                {"type": "Ball Possession", "value": "45%"},
                {"type": "Yellow Cards", "value": 3},
                {"type": "Total passes", "value": 380},
                {"type": "Passes accurate", "value": 310},
                {"type": "Passes %", "value": "82%"}
            ]
        }
    ]
    
    return {
        "fixture": simulated_fixture,
        "statistics": simulated_stats
    }



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
    client = APIFootballClient()
    ingested_count = 0
    
    try:
        raw_fixtures = client.fetch_next_fixtures(league_id, season, next_n)
        
        for raw_fixture in raw_fixtures:
            try:
                # Clean and insert match data
                cleaned = clean_fixture_data(raw_fixture)
                
                # Fetch odds
                fixture_id = cleaned.get('match_id')
                if fixture_id:
                    odds = client.fetch_odds(fixture_id)
                    if odds:
                        cleaned.update(odds)
                        logger.info(f"  > Odds found: {odds}")
                
                upsert_match(cleaned)
                ingested_count += 1
                logger.info(f"Ingested upcoming: {cleaned['home_team']} vs {cleaned['away_team']}")
                
            except Exception as e:
                logger.error(f"Error ingesting upcoming fixture: {e}")
                continue
        
        logger.info(f"Successfully ingested {ingested_count} upcoming fixtures")
        return ingested_count
        
    except Exception as e:
        logger.error(f"Error during upcoming ingestion: {e}")
        raise


def ingest_all_leagues(fetch_stats: bool = True) -> Dict[str, int]:
    """
    Ingest fixtures for ALL configured leagues in LEAGUE_IDS.
    Loops over each league and stores everything in the same matches table.
    
    Args:
        fetch_stats: Whether to fetch detailed statistics for finished matches
        
    Returns:
        Dict mapping league_name -> number of fixtures ingested
    """
    results: Dict[str, int] = {}
    total = 0
    
    for league_id, info in LEAGUE_IDS.items():
        name = str(info["name"])
        season = int(info["season"])
        
        logger.info(f"{'='*50}")
        logger.info(f"Ingesting: {name} (ID={league_id}, season={season})")
        logger.info(f"{'='*50}")
        
        try:
            count = ingest_fixtures(league_id, season, fetch_stats=fetch_stats)
            results[name] = count
            total += count
            logger.info(f"✅ {name}: {count} fixtures ingested")
        except Exception as e:
            logger.error(f"❌ {name}: ingestion failed — {e}")
            results[name] = 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TOTAL: {total} fixtures ingested across {len(LEAGUE_IDS)} leagues")
    for name, count in results.items():
        logger.info(f"  {name}: {count}")
    
    return results

