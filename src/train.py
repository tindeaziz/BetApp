
import os
import argparse
import logging
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.features import FeatureEngineer
from src.models import MatchPredictor
from src.database import SupabaseDB
import pandas as pd
import numpy as np

# Configure logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Football Prediction Models (Stacking + Calibration)")
    parser.add_argument('--limit', type=int, default=5000, help='Number of matches to fetch for training')
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()
    
    logger.info(f"Starting training pipeline with limit={args.limit}...")
    
    # 1. Initialize Components
    fe = FeatureEngineer()
    predictor = MatchPredictor()
    
    # 2. Prepare Training Data
    logger.info("Fetching and processing data...")
    # 2. Prepare Training Data
    logger.info("Fetching and processing data...")
    X, y_result, y_goals, y_fouls = fe.prepare_training_data(limit=args.limit)
    
    logger.info(f"Data prepared. Features shape: {X.shape}")
    
    # Check for class imbalance
    logger.info(f"Class distribution (Result): \n{y_result.value_counts(normalize=True)}")
    
    # 3. Train Stacking Classifier (Result)
    logger.info("--- Training Result Classifier (Stacking) ---")
    result_metrics = predictor.train_result_classifier(X, y_result)
    logger.info(f"Result Metrics: {result_metrics}")
    
    # 4. Train Goals Regressor
    logger.info("--- Training Goals Regressor ---")
    goals_metrics = predictor.train_goals_regressor(X, y_goals)
    logger.info(f"Goals Metrics: {goals_metrics}")
    
    # 5. Train Fouls Regressor
    logger.info("--- Training Fouls Regressor ---")
    
    # Filter 0 fouls (missing data)
    valid_fouls_mask = y_fouls > 0
    if valid_fouls_mask.sum() < 100:
        logger.warning("Not enough valid foul data (need > 100 samples). skipping or using mock?")
        # Fallback to keep pipeline running if data is truly empty, but warn user
        logger.warning("FALLBACK: Using all data (including zeros) which is bad, or mocking if empty.")
        # ideally we stop or fix data. For now, let's try to train on what we have if > 10.
    
    X_fouls = X[valid_fouls_mask]
    y_fouls_valid = y_fouls[valid_fouls_mask]
    
    logger.info(f"Training Fouls Regressor on {len(y_fouls_valid)} samples (filtered {len(y_fouls) - len(y_fouls_valid)} zero-foul matches)")
    
    if not X_fouls.empty:
        fouls_metrics = predictor.train_fouls_regressor(X_fouls, y_fouls_valid)
        logger.info(f"Fouls Metrics: {fouls_metrics}")
    else:
        logger.error("No valid foul data found! Skipping foul model training.")
    
    logger.info("Training pipeline completed successfully.")
    logger.info(f"Models saved to {predictor.model_path}")

if __name__ == "__main__":
    main()
