"""
Hyperparameter Optimization for XGBoost Result Classifier using Optuna.

Usage:
    python -m src.optimize                  # 50 trials (default)
    python -m src.optimize --trials 100     # custom trial count
    python -m src.optimize --limit 2000     # more training data
"""

import argparse
import json
import logging
import os
from typing import Dict, Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.features import FeatureEngineer

# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logs (keep only important info)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────
def load_training_data(limit: int = 1000):
    """
    Load and prepare training data for the result classifier.

    Returns:
        X: Feature DataFrame (22 features)
        y: Encoded target array (0=A, 1=D, 2=H)
        le: Fitted LabelEncoder
    """
    logger.info(f"Loading training data (limit={limit})...")
    fe = FeatureEngineer()
    X, y_result, _ = fe.prepare_training_data(limit=limit)

    le = LabelEncoder()
    y = le.fit_transform(y_result)

    logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features, classes={list(le.classes_)}")
    return X, y, le


# ─────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────
def objective(trial: optuna.Trial, X, y) -> float:
    """
    Optuna objective function.
    Suggests hyperparameters, trains XGBoost with 5-fold CV,
    and returns mean accuracy.
    """

    # ── Hyperparameter search space ──
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # ── Build model ──
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        **params,
    )

    # ── 5-fold stratified cross-validation ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return scores.mean()


# ─────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────
def run_optimization(n_trials: int = 50, data_limit: int = 1000) -> Dict[str, Any]:
    """
    Run the full Optuna optimization loop.

    Args:
        n_trials:   Number of Optuna trials
        data_limit: Max matches for training data

    Returns:
        Dictionary with best_params, best_accuracy, and study summary
    """
    X, y, le = load_training_data(limit=data_limit)

    logger.info(f"Starting Optuna optimization — {n_trials} trials")
    logger.info("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_result_classifier",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Pass data to objective via lambda
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, show_progress_bar=True)

    # ── Results ──
    best = study.best_trial

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best Accuracy (CV): {best.value:.4f}")
    logger.info(f"  Best Trial:         #{best.number}")
    logger.info("")
    logger.info("  Best Params:")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")

    # ── Save best params to JSON ──
    output_path = os.path.join("models", "best_params.json")
    os.makedirs("models", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"best_accuracy_cv": best.value, "best_params": best.params},
            f,
            indent=2,
        )
    logger.info(f"\n  Best params saved to: {output_path}")

    # ── Top 5 trials ──
    logger.info("\n  Top 5 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for t in top_trials:
        logger.info(f"    Trial #{t.number}: accuracy={t.value:.4f}")

    return {
        "best_accuracy_cv": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
    }


# ─────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for XGBoost")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--limit", type=int, default=1000, help="Training data limit (default: 1000)")
    args = parser.parse_args()

    results = run_optimization(n_trials=args.trials, data_limit=args.limit)

    print("\n" + "=" * 60)
    print("  COPY THESE INTO src/models.py train_result_classifier():")
    print("=" * 60)
    print(f"""
    self.result_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        learning_rate={results['best_params']['learning_rate']:.6f},
        max_depth={results['best_params']['max_depth']},
        n_estimators={results['best_params']['n_estimators']},
        subsample={results['best_params']['subsample']:.4f},
        colsample_bytree={results['best_params']['colsample_bytree']:.4f},
        min_child_weight={results['best_params']['min_child_weight']},
        gamma={results['best_params']['gamma']:.4f},
        reg_alpha={results['best_params']['reg_alpha']:.8f},
        reg_lambda={results['best_params']['reg_lambda']:.8f},
        random_state=42,
        eval_metric='mlogloss'
    )""")
    print("=" * 60)
