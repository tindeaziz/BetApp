
import os
import joblib
import numpy as np
import logging
from dotenv import load_dotenv
from src.features import FeatureEngineer

# Load environment
load_dotenv()
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/")

FEATURE_COLUMNS = [
    'home_form', 'away_form',
    'home_offensive', 'away_offensive',
    'home_defensive', 'away_defensive',
    'offensive_diff', 'defensive_diff', 'form_diff',
    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_avg_goals',
    'referee_aggression', 'home_advantage',
    # Advanced features
    'home_elo', 'away_elo',
    'home_attack_strength', 'away_defense_strength',
    'is_derby',
    # Competition features
    'competition_type', 'is_knockout'
]

def load_models():
    """Load ML models from disk."""
    models = {}
    files = {
        'result_classifier': 'result_classifier.joblib',
        'goals_regressor': 'goals_regressor.joblib',
        'fouls_regressor': 'fouls_regressor.joblib',
        'label_encoder': 'label_encoder.joblib'
    }
    
    # Try to load stacking model first for result classifier
    stacking_path = os.path.join(MODEL_PATH, "result_stacking.joblib")
    if os.path.exists(stacking_path):
        models['result_classifier'] = joblib.load(stacking_path)
    
    for key, filename in files.items():
        if key == 'result_classifier' and key in models:
            continue # Already loaded stacking
            
        filepath = os.path.join(MODEL_PATH, filename)
        if not os.path.exists(filepath):
            if key == 'result_classifier': # Critical
                return None
            continue
        models[key] = joblib.load(filepath)
    
    return models

def get_feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()

def run_prediction(home_team: str, away_team: str, models: dict = None, referee: str | None = None,
                   league: str | None = None, league_id: int | None = None,
                   match_date: str | None = None) -> dict:
    """Run prediction for a match and return results."""
    if models is None:
        models = load_models()
    
    if models is None:
        raise ValueError("Models could not be loaded")

    fe = get_feature_engineer()
    
    # Build features (same order as training)
    features_df = fe.create_match_features(
        home_team, away_team, referee,
        league=league, league_id=league_id, match_date=match_date
    )
    # Preserve full features for UI display before filtering for model
    full_features = features_df.copy()
    
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[FEATURE_COLUMNS]
    

    # Predict
    result_proba = models['result_classifier'].predict_proba(features_df)[0]
    result_classes = models['label_encoder'].classes_
    prob_dict = dict(zip(result_classes, result_proba))
    
    predicted_idx = np.argmax(result_proba)
    predicted_result = result_classes[predicted_idx]
    confidence = float(result_proba[predicted_idx])
    
    predicted_goals = max(0.0, float(models['goals_regressor'].predict(features_df)[0]))
    predicted_fouls = max(0, int(models['fouls_regressor'].predict(features_df)[0]))
    
    # Normalization (H+D+A = 1.0)
    raw_h, raw_d, raw_a = prob_dict.get('H', 0), prob_dict.get('D', 0), prob_dict.get('A', 0)
    total = raw_h + raw_d + raw_a
    if total > 0:
        prob_h, prob_d, prob_a = raw_h/total, raw_d/total, raw_a/total
    else:
        prob_h, prob_d, prob_a = 0.33, 0.34, 0.33

    # --- NEW ANALYTICS INTEGRATION ---
    from src.analytics import (
        calculate_expected_goals, 
        calculate_correct_score_probs, 
        calculate_btts_prob, 
        calculate_over_under_probs,
        detect_value_bet
    )
    from src.foul_engine import (
        get_referee_strictness, 
        calculate_match_intensity, 
        predict_fouls_advanced, 
        predict_cards
    )

    # 1. Advanced Goal Analytics (Poisson)
    # Estimate team strengths from features or probabilities
    # Using 'home_offensive' feature if available, else heuristic from win probs
    h_strength = features_df.get('home_offensive', [1.0]).iloc[0]
    a_strength = features_df.get('away_offensive', [1.0]).iloc[0]
    
    home_xg, away_xg = calculate_expected_goals(predicted_goals, h_strength, a_strength)
    
    # Correct Score & BTTS
    correct_scores = calculate_correct_score_probs(home_xg, away_xg)
    btts_prob = calculate_btts_prob(home_xg, away_xg)
    over_under_probs = calculate_over_under_probs(home_xg, away_xg)
    
    # 2. Advanced Foul Analytics
    # Mock referee stats (would come from DB in real implementation)
    # We use 'referee_aggression' feature as a proxy for strictness
    ref_aggression_feature = features_df.get('referee_aggression', [3.5]).iloc[0]
    # Normalize: feature is roughly fouls/card weight. We map it to ~ 0.8-1.5 range
    ref_strictness = ref_aggression_feature / 5.0 # Crude approximation based on feature scale
    
    is_derby = bool(features_df.get('is_derby', [0]).iloc[0])
    match_intensity = calculate_match_intensity(is_derby, importance_score=0.5, history_fouls_avg=25)
    
    final_fouls = predict_fouls_advanced(predicted_fouls, ref_strictness, match_intensity)
    card_analytics = predict_cards(final_fouls, referee_card_rate=0.15)
    
    # 3. Value Bet Detection (Input odds if we had them, defaulting to 1.0)
    # in real app, 'odds' would be passed to run_prediction
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'prob_home': prob_h,
        'prob_draw': prob_d,
        'prob_away': prob_a,
        'prob_1x': prob_h + prob_d,
        'prob_x2': prob_a + prob_d,
        'prob_12': prob_h + prob_a,
        'predicted_result': predicted_result,
        'confidence': confidence,
        'predicted_goals': predicted_goals,
        'predicted_fouls': final_fouls, # Use the advanced prediction
        # New Metrics
        'home_xg': home_xg,
        'away_xg': away_xg,
        'correct_score_probs': correct_scores,
        'btts_prob': btts_prob,
        'over_under_probs': over_under_probs,
        'foul_analytics': {
            'intensity_score': match_intensity,
            'referee_strictness': ref_strictness,
            'expected_cards': card_analytics['expected_cards'],
            'prob_over_3_5_cards': card_analytics['prob_over_3_5']
        },
        'features': full_features.iloc[0].to_dict(),
        'explanation': generate_ai_explanation(
            home_team, away_team, 
            prob_h, prob_a, 
            full_features.iloc[0].to_dict()
        )
    }

def generate_ai_explanation(home, away, prob_h, prob_a, feats):
    """Generate natural language explanation based on features."""
    reasons = []
    
    # Delta thresholds
    SIGNIFICANT_DIFF = 0.15
    
    # 1. Form Analysis
    h_form = feats.get('home_form', 0)
    a_form = feats.get('away_form', 0)
    diff_form = h_form - a_form
    
    if diff_form > SIGNIFICANT_DIFF:
        reasons.append(f"<b>{home}</b> arrives in much better shape, with a superior recent form rating ({h_form:.1f} vs {a_form:.1f}).")
    elif diff_form < -SIGNIFICANT_DIFF:
        reasons.append(f"<b>{away}</b> has shown better momentum recently, outperforming {home} in form ({a_form:.1f} vs {h_form:.1f}).")
    
    # 2. Offensive/Defensive Matchups
    h_att = feats.get('home_offensive', 0)
    a_def = feats.get('away_defensive', 0)
    a_att = feats.get('away_offensive', 0)
    h_def = feats.get('home_defensive', 0)
    
    if h_att > a_def + 0.5:
        reasons.append(f"The home attack is performing well above {away}'s defensive average.")
        
    # 3. Head to Head
    h2h_home = feats.get('h2h_home_wins', 0)
    h2h_away = feats.get('h2h_away_wins', 0)
    
    if h2h_home > 0.6:
        reasons.append(f"Historically, <b>{home}</b> has dominated this fixture.")
    elif h2h_away > 0.6:
        reasons.append(f"<b>{away}</b> has a strong psychological advantage based on past meetings.")
        
    # 4. Probability Context
    if prob_h > 0.6:
        base = f"The model strongly favors <b>{home}</b>."
    elif prob_a > 0.6:
        base = f"The model strongly favors <b>{away}</b>."
    elif abs(prob_h - prob_a) < 0.1:
        base = "This is a tightly contested match with no clear favorite."
    else:
        fav = home if prob_h > prob_a else away
        base = f"<b>{fav}</b> has a slight edge in this matchup."
        
    # Combine
    if not reasons:
        text = f"{base} Both teams appear evenly matched in key metrics."
    else:
        text = f"{base} {' '.join(reasons)}"
        
    return text

