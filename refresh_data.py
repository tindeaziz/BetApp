
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import ingest_upcoming_fixtures, ingest_fixtures, LEAGUE_IDS
from datetime import datetime
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def refresh_data():
    logger.info("Starting data refresh to fetch missing logos...")
    
    # Refresh upcoming fixtures for major leagues
    for league_id, info in LEAGUE_IDS.items():
        try:
            logger.info(f"Checking {info['name']}...")
            ingest_upcoming_fixtures(league_id, info['season'], next_n=5)
            # Also check today's fixtures specifically if any
            today = datetime.now().strftime("%Y-%m-%d")
            ingest_fixtures(league_id, info['season'], date_from=today, date_to=today, fetch_stats=False)
            
        except Exception as e:
            logger.error(f"Error refreshing {info['name']}: {e}")
            
    logger.info("\n✅ Data refresh complete. Please restart the app or click Refresh.")

if __name__ == "__main__":
    refresh_data()
