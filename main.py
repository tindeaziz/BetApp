"""
Main entry point for the Football Match Prediction Application.
Provides CLI interface for data ingestion, model training, and predictions.
"""

import argparse
import sys
import logging
from typing import Optional

from src.database import init_db
from src.ingestion import ingest_fixtures, simulate_fetch_data
from src.models import MatchPredictor, train_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database() -> None:
    """Initialize database schema."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


def run_ingestion(
    league_id: int,
    season: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> None:
    """
    Run data ingestion from API-Football.
    
    Args:
        league_id: League ID
        season: Season year
        date_from: Start date (optional)
        date_to: End date (optional)
    """
    logger.info(f"Starting data ingestion for league {league_id}, season {season}")
    
    try:
        count = ingest_fixtures(league_id, season, date_from, date_to)
        logger.info(f"Successfully ingested {count} fixtures")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


def run_training(data_limit: int = 1000) -> None:
    """
    Train all ML models.
    
    Args:
        data_limit: Maximum number of matches to use for training
    """
    logger.info("Starting model training...")
    
    try:
        metrics = train_all_models(data_limit=data_limit)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE - MODEL METRICS")
        logger.info("="*50)
        
        logger.info("\nResult Classifier (1N2):")
        logger.info(f"  Accuracy: {metrics['result_classifier']['accuracy']:.4f}")
        logger.info(f"  CV Score: {metrics['result_classifier']['cv_mean']:.4f} "
                   f"(+/- {metrics['result_classifier']['cv_std']:.4f})")
        
        logger.info("\nGoals Regressor (Over/Under):")
        logger.info(f"  RMSE: {metrics['goals_regressor']['rmse']:.4f}")
        logger.info(f"  R²: {metrics['goals_regressor']['r2']:.4f}")
        
        logger.info("\nFouls Regressor:")
        logger.info(f"  RMSE: {metrics['fouls_regressor']['rmse']:.4f}")
        logger.info(f"  R²: {metrics['fouls_regressor']['r2']:.4f}")
        
        logger.info("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def run_prediction(
    home_team: str,
    away_team: str,
    referee: Optional[str] = None
) -> None:
    """
    Predict match outcome.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        referee: Referee name (optional)
    """
    logger.info(f"Predicting match: {home_team} vs {away_team}")
    
    try:
        predictor = MatchPredictor()
        predictor.load_models()
        
        prediction = predictor.predict_match(home_team, away_team, referee)
        
        # Display prediction
        logger.info("\n" + "="*50)
        logger.info("MATCH PREDICTION")
        logger.info("="*50)
        logger.info(f"Match: {home_team} vs {away_team}")
        if referee:
            logger.info(f"Referee: {referee}")
        logger.info("\nResult Probabilities:")
        logger.info(f"  Home Win: {prediction['prob_home_win']:.2%}")
        logger.info(f"  Draw: {prediction['prob_draw']:.2%}")
        logger.info(f"  Away Win: {prediction['prob_away_win']:.2%}")
        logger.info(f"\nPredicted Result: {prediction['predicted_result']}")
        logger.info(f"Confidence: {prediction['confidence_score']:.2%}")
        logger.info(f"\nExpected Total Goals: {prediction['predicted_over_under']:.2f}")
        logger.info(f"Expected Total Fouls: {prediction['predicted_fouls']}")
        logger.info("="*50 + "\n")
        
    except FileNotFoundError:
        logger.error("Models not found. Please train models first using: python main.py train")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Football Match Prediction Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database
  python main.py init

  # Ingest data for Premier League 2023 season
  python main.py ingest --league 39 --season 2023

  # Train models
  python main.py train --limit 1000

  # Predict a match
  python main.py predict --home "Manchester United" --away "Liverpool"
  python main.py predict --home "Manchester United" --away "Liverpool" --referee "Michael Oliver"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    subparsers.add_parser('init', help='Initialize database schema')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest match data from API')
    ingest_parser.add_argument('--league', type=int, required=True, help='League ID')
    ingest_parser.add_argument('--season', type=int, required=True, help='Season year')
    ingest_parser.add_argument('--from', dest='date_from', help='Start date (YYYY-MM-DD)')
    ingest_parser.add_argument('--to', dest='date_to', help='End date (YYYY-MM-DD)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--limit', type=int, default=1000, 
                             help='Maximum number of matches for training (default: 1000)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict match outcome')
    predict_parser.add_argument('--home', required=True, help='Home team name')
    predict_parser.add_argument('--away', required=True, help='Away team name')
    predict_parser.add_argument('--referee', help='Referee name (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    if args.command == 'init':
        setup_database()
    
    elif args.command == 'ingest':
        run_ingestion(args.league, args.season, args.date_from, args.date_to)
    
    elif args.command == 'train':
        run_training(args.limit)
    
    elif args.command == 'predict':
        run_prediction(args.home, args.away, args.referee)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
