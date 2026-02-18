"""
🔄 Daily Prediction Batch Script
================================

This script is designed to be run automatically (e.g., via GitHub Actions)
early in the morning. It:
1. Fetches all matches scheduled for TODAY.
2. Checks if a prediction already exists.
3. Generates a prediction using the trained XGBoost models.
4. Saves the prediction to the Supabase database.

Usage:
    python daily_predictions.py
"""

import os
import sys
import logging
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import get_todays_matches, get_cached_prediction, save_prediction_cache
from src.predict import load_brains, predict_match
from src.features import FeatureEngineer

def run_daily_predictions():
    """Main execution function."""
    logger.info("🚀 Starting Daily Prediction Batch...")
    
    # 1. Load Models
    try:
        models = load_brains()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)
        
    # 2. Initialize Feature Engineer (DB Connection)
    try:
        fe = FeatureEngineer()
    except Exception as e:
        logger.error(f"Failed to initialize FeatureEngineer: {e}")
        sys.exit(1)

    # 3. Get Today's Matches
    matches = get_todays_matches()
    if not matches:
        logger.info("😴 No matches scheduled for today. Exiting.")
        return

    logger.info(f"📅 Found {len(matches)} matches for today. Processing...")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for match in matches:
        match_id = match['id']
        home_team = match['home']
        away_team = match['away']
        
        logger.info(f"🔍 Processing: {home_team} vs {away_team} (ID: {match_id})")

        # Check cache first
        existing = get_cached_prediction(match_id)
        if existing:
            logger.info("   ⏭️  Prediction already exists. Skipping.")
            skipped_count += 1
            continue

        try:
            # Generate Prediction
            prediction = predict_match(
                home_team=home_team,
                away_team=away_team,
                models=models,
                fe=fe,
                referee=match.get('referee')
            )
            
            # Add match_id to prediction dict for saving
            # (predict_match returns a dict, but save_prediction_cache needs match_id separately or in the dict)
            # checking src/database.py save_prediction_cache signature: (match_id, prediction)
            
            save_prediction_cache(match_id, prediction)
            processed_count += 1
            logger.info("   ✅ Prediction saved.")
            
        except Exception as e:
            logger.error(f"   ❌ Error predicting match {match_id}: {e}")
            error_count += 1

    logger.info("="*50)
    logger.info("🏁 Batch Completed")
    logger.info(f"   Processed: {processed_count}")
    logger.info(f"   Skipped:   {skipped_count}")
    logger.info(f"   Errors:    {error_count}")
    logger.info("="*50)

if __name__ == "__main__":
    run_daily_predictions()
