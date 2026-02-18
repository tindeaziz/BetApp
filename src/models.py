
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import logging
import joblib



from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, log_loss
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import xgboost as xgb

# lightgbm and catboost removed due to Python 3.14 compatibility issues
# import lightgbm as lgb
# from catboost import CatBoostClassifier

from src.features import FeatureEngineer
from src.database import insert_prediction, SupabaseDB

# Configure logging
# Configure logging
logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Advanced predictor class for football match outcomes.
    Uses Stacking (XGB+HistGB+RF) with Probability Calibration for 1N2 predictions.
    Uses XGBoost for regression tasks (Goals, Fouls).
    """
    
    def __init__(self, model_path: str = "models/"):
        """
        Initialize the match predictor.
        """
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        
        # Models
        self.result_pipeline: Optional[Any] = None # Stacking + Calibration
        self.goals_regressor: Optional[xgb.XGBRegressor] = None
        self.fouls_regressor: Optional[xgb.XGBRegressor] = None
        
        # Label encoder for results
        self.label_encoder = LabelEncoder()
        
        # Model version
        self.model_version = "v2.0-stacking-calibrated"
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
    
    def build_stacking_pipeline(self, random_state: int = 42) -> Any:
        """
        Build the Stacking Classifier pipeline with Calibration.
        
        Structure:
        1. Base Learners: 
           - XGBoost (optimized)
           - HistGradientBoostingClassifier (LightGBM alternative)
           - RandomForestClassifier (diversity)
        2. Meta Learner: Logistic Regression
        3. Calibration: Isotonic Regression
        """
        # Base Learners
        estimators = [
            ('xgb', xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )),
            ('hist_gb', HistGradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.05,
                max_iter=300,
                max_depth=6,
                random_state=random_state,
                early_stopping=False
            )),
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            ))
        ]
        
        # Stacking
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=-1,
            passthrough=False
        )
        
        # Calibration
        calibrated_clf = CalibratedClassifierCV(
            estimator=stacking_clf,
            method='isotonic',
            cv=3
        )
        
        return calibrated_clf

    def train_result_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the Stacking Ensemble for match result prediction.
        """
        logger.info(f"Training Stacking Ensemble (XGB+LGBM+CatBoost) with Calibration on {len(X)} samples")
        
        try:
            # Ensure y is string type
            y = y.astype(str)
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            # Build and Train Pipeline
            self.result_pipeline = self.build_stacking_pipeline(random_state)
            
            logger.info("Fitting model pipeline (this may take a while)...")
            self.result_pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.result_pipeline.predict(X_test)
            y_pred_proba = self.result_pipeline.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            logger.info(f"Stacked Classifier - Accuracy: {accuracy:.4f}, LogLoss: {logloss:.4f}")
            
            # Classification report
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            report = classification_report(y_test_labels, y_pred_labels)
            logger.info(f"\nClassification Report:\n{report}")
            
            # Save model
            model_file = os.path.join(self.model_path, "result_stacking.joblib")
            joblib.dump(self.result_pipeline, model_file)
            
            # Save label encoder
            encoder_file = os.path.join(self.model_path, "label_encoder.joblib")
            joblib.dump(self.label_encoder, encoder_file)
            
            logger.info(f"Model saved to {model_file}")
            
            return {
                'accuracy': accuracy,
                'log_loss': logloss,
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error training result classifier: {e}")
            raise
    
    def train_goals_regressor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train XGBoost regressor for total goals prediction (Over/Under).
        """
        logger.info(f"Training goals regressor on {len(X)} samples")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize XGBoost regressor
            self.goals_regressor = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=5,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Train model
            self.goals_regressor.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred = self.goals_regressor.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Goals Regressor - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Save model
            model_file = os.path.join(self.model_path, "goals_regressor.joblib")
            joblib.dump(self.goals_regressor, model_file)
            
            return {'rmse': rmse, 'r2': r2}
            
        except Exception as e:
            logger.error(f"Error training goals regressor: {e}")
            raise
    
    def train_fouls_regressor(self, X, y_fouls, test_size=0.2, random_state=42):
        """Train XGBoost regressor for fouls prediction."""
        # Simplified wrapper to keep interface consistent
        logger.info("Training fouls regressor...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_fouls, test_size=test_size, random_state=random_state)
        self.fouls_regressor = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
        self.fouls_regressor.fit(X_train, y_train)
        y_pred = self.fouls_regressor.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        joblib.dump(self.fouls_regressor, os.path.join(self.model_path, "fouls_regressor.joblib"))
        logger.info(f"Fouls Regressor RMSE: {rmse:.4f}")
        return {'rmse': rmse}

    def load_models(self) -> None:
        """
        Load trained models from disk.
        """
        try:
            # Load Stacking Pipeline
            stacking_file = os.path.join(self.model_path, "result_stacking.joblib")
            # Fallback to old classifier if stacking not found? No, better fail or retrain.
            if os.path.exists(stacking_file):
                self.result_pipeline = joblib.load(stacking_file)
                logger.info("Loaded Stacking Ensemble")
            else:
                # Try loading old XGBoost for backward compatibility/during transition
                xgb_file = os.path.join(self.model_path, "result_classifier.joblib")
                if os.path.exists(xgb_file):
                    self.result_pipeline = joblib.load(xgb_file)
                    logger.warning("Loaded Legacy XGBoost Classifier (Stacking not found)")
                else:
                    raise FileNotFoundError("No result classifier found.")

            goals_file = os.path.join(self.model_path, "goals_regressor.joblib")
            fouls_file = os.path.join(self.model_path, "fouls_regressor.joblib")
            encoder_file = os.path.join(self.model_path, "label_encoder.joblib")
            
            self.goals_regressor = joblib.load(goals_file)
            self.fouls_regressor = joblib.load(fouls_file)
            self.label_encoder = joblib.load(encoder_file)
            
            logger.info("All regression models loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        referee: str = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Predict match outcome using the loaded pipeline.
        """
        if self.result_pipeline is None or self.goals_regressor is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        logger.info(f"Predicting match: {home_team} vs {away_team}")
        
        try:
            # Create features
            features = self.feature_engineer.create_match_features(
                home_team, away_team, referee
            )
            
            # Predict result probabilities
            # The pipeline (CalibratedClassifierCV) creates robust probabilities
            result_proba = self.result_pipeline.predict_proba(features)[0]
            result_classes = self.label_encoder.classes_
            
            # Map probabilities to classes
            prob_dict = dict(zip(result_classes, result_proba))
            
            # Predicted result
            predicted_idx = np.argmax(result_proba)
            predicted_result = result_classes[predicted_idx]
            
            # Predict goals
            predicted_goals = self.goals_regressor.predict(features)[0]
            
            # Predict fouls
            predicted_fouls = self.fouls_regressor.predict(features)[0]
            
            # Confidence score
            confidence = float(np.max(result_proba))
            
            # Extract raw probs
            raw_h = float(prob_dict.get('H', 0.0))
            raw_d = float(prob_dict.get('D', 0.0))
            raw_a = float(prob_dict.get('A', 0.0))
            
            total_prob = raw_h + raw_d + raw_a
            
            # Integrity Check
            if total_prob < 0.99:
                logger.warning(f"⚠️ Probability sum mismatch: {total_prob:.4f} (Missing mass: {1-total_prob:.4f}). Renormalizing...")
                # Normalize
                if total_prob > 0:
                    prob_h = raw_h / total_prob
                    prob_d = raw_d / total_prob
                    prob_a = raw_a / total_prob
                else:
                    # Fallback if total is 0 (unlikely)
                    prob_h, prob_d, prob_a = 0.33, 0.34, 0.33
            else:
                prob_h, prob_d, prob_a = raw_h, raw_d, raw_a

            prediction = {
                'home_team': home_team,
                'away_team': away_team,
                'prob_home_win': prob_h,
                'prob_draw': prob_d,
                'prob_away_win': prob_a,
                'predicted_result': predicted_result,
                'predicted_over_under': float(predicted_goals),
                'predicted_fouls': int(predicted_fouls),
                'confidence_score': max(prob_h, prob_d, prob_a), # Recalculate confidence based on new probs
                'model_version': self.model_version,
                'prediction_date': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction: {predicted_result} (Conf: {prediction['confidence_score']:.2%})")
            
            if save_to_db:
                try:
                    insert_prediction(prediction)
                except Exception as e:
                    logger.warning(f"Failed to save prediction: {e}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            raise

# Note: train_all_models is removed from here as it will be orchestrated by src/train.py
