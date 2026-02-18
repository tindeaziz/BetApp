
"""
Analytics module for Poisson-based probabilities (Correct Score, BTTS, Over/Under).
"""
import numpy as np
from scipy.stats import poisson
from typing import Dict, Any, Tuple

def calculate_expected_goals(total_xg: float, home_strength: float, away_strength: float) -> Tuple[float, float]:
    """
    Split total expected goals into Home and Away xG based on relative strength.
    
    Args:
        total_xg: Total expected goals for the match (from regression model).
        home_strength: Home team attack/defense factor (approximate).
        away_strength: Away team attack/defense factor.
    
    Returns:
        (home_xg, away_xg)
    """
    # Simple heuristic splitting if we don't have individual team xG models.
    # We assume strength features are roughly centered around 1.0 or similar.
    # If we don't have explicit xG per team, we infer from probabilities or strength features.
    
    # Heuristic: Ratio of strengths
    # Avoid division by zero
    if home_strength + away_strength == 0:
        ratio = 0.5
    else:
        ratio = home_strength / (home_strength + away_strength)
        
    home_xg = total_xg * ratio
    away_xg = total_xg * (1 - ratio)
    
    return home_xg, away_xg

def calculate_correct_score_probs(home_xg: float, away_xg: float, max_goals: int = 5) -> Dict[str, float]:
    """
    Calculate correct score probabilities using independent Poisson distributions.
    
    Returns:
        Dict mapped as "Home-Away": probability (e.g. "1-0": 0.12)
    """
    probs = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson.pmf(h, home_xg)
            p_a = poisson.pmf(a, away_xg)
            score = f"{h}-{a}"
            probs[score] = p_h * p_a
            
    # Normalize to ensure sum is closer to 1 (truncation at max_goals loses some mass)
    total_prob = sum(probs.values())
    if total_prob > 0:
        for k in probs:
            probs[k] /= total_prob
            
    # Sort by probability descending
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

def calculate_btts_prob(home_xg: float, away_xg: float) -> float:
    """
    Calculate probability of Both Teams To Score (BTTS).
    P(BTTS) = (1 - P(Home=0)) * (1 - P(Away=0))
    """
    p_home_score = 1 - poisson.pmf(0, home_xg)
    p_away_score = 1 - poisson.pmf(0, away_xg)
    
    return p_home_score * p_away_score

def calculate_over_under_probs(home_xg: float, away_xg: float) -> Dict[str, float]:
    """
    Calculate Over/Under 2.5 probabilities.
    """
    under_2_5 = 0.0
    
    # Sum probabilities for all scores where h+a < 2.5 (i.e. 0, 1, 2 goals total)
    for h in range(4):
        for a in range(4):
            if h + a < 2.5:
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                under_2_5 += p
                
    return {
        "over_2_5": 1.0 - under_2_5,
        "under_2_5": under_2_5
    }

def detect_value_bet(model_prob: float, bookie_odds: float, threshold: float = 0.05) -> bool:
    """
    Identify if there is value in the bet.
    Value = (ModelProb * Odds) - 1
    """
    if bookie_odds <= 1.0: return False
    
    expected_value = (model_prob * bookie_odds) - 1.0
    return expected_value > threshold
