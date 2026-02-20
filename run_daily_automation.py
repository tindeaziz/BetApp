import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.ingestion import ingest_all_supported_leagues
from src.database import get_todays_matches, get_cached_prediction, save_prediction_cache
from src.prediction import load_models, run_prediction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

def run_daily_automation():
    """
    Daily automation job that:
    1. Fetches fixtures and odds for the next 2 days.
    2. Runs the AI models on all upcoming matches.
    3. Caches the predictions in the database.
    """
    logger.info("==================================================")
    logger.info("🚀 STARTING DAILY AUTOMATION JOB")
    logger.info("==================================================")
    
    # Step 1: Ingest new fixtures and odds for today and tomorrow
    logger.info("STEP 1: Fetching new fixtures and odds...")
    try:
        # We fetch a few days ahead to be safe (e.g. 2-3 days)
        matches_ingested = ingest_all_supported_leagues(days_ahead=2)
        logger.info(f"✅ Successfully ingested {matches_ingested} upcoming matches.")
    except Exception as e:
        logger.error(f"❌ Failed to ingest fixtures: {e}")
        return

    # Step 2: Load Models
    logger.info("STEP 2: Loading AI Models...")
    try:
        models = load_models()
        if not models:
            logger.error("❌ Failed to load AI models. Make sure they are trained.")
            return
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return

    # Step 3: Run Predictions and Cache them
    logger.info("STEP 3: Generating and Caching Predictions...")
    try:
        # Get all upcoming matches (today)
        todays_matches = get_todays_matches()
        logger.info(f"Found {len(todays_matches)} matches to predict today.")
        
        predicted_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, match in enumerate(todays_matches):
            match_id = match['id']
            home = match['home']
            away = match['away']
            
            logger.info(f"Processing ({idx+1}/{len(todays_matches)}): {home} vs {away}")
            
            # Check if prediction is already cached
            cached = get_cached_prediction(match_id)
            if cached and 'features' in cached.get('features', {}):
                logger.info(f"  ⏭️ Prediction already cached. Skipping.")
                skipped_count += 1
                continue
                
            # Run prediction
            try:
                pred = run_prediction(
                    home, away,
                    models=models,
                    referee=match.get('referee'),
                    league=match.get('league'),
                    league_id=match.get('league_id'),
                    match_date=match.get('match_date')
                )
                
                # Save to cache
                save_prediction_cache(match_id, pred)
                logger.info(f"  ✅ Prediction generated and cached: {pred['predicted_result']}")
                predicted_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ Error predicting {home} vs {away}: {e}")
                error_count += 1
                
        logger.info("==================================================")
        logger.info(f"🏁 AUTOMATION COMPLETE")
        logger.info(f"   Newly Predicted: {predicted_count}")
        logger.info(f"   Skipped (Cached): {skipped_count}")
        logger.info(f"   Errors: {error_count}")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"❌ Automation pipeline failed during prediction phase: {e}")


if __name__ == "__main__":
    run_daily_automation()
