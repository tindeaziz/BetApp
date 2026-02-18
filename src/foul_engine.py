
"""
Foul Engine: Advanced prediction logic for Fouls and Cards.
Incorporates Referee Strictness and Match Intensity.
"""
import numpy as np
from typing import Dict, Any

def get_referee_strictness(referee_stats: Dict[str, Any]) -> float:
    """
    Calculate a multiplier for referee strictness.
    Baseline is 1.0.
    """
    if not referee_stats:
        return 1.0
        
    # Example stats: avg_fouls_per_game, avg_cards_per_game
    # Global averages (heuristics)
    AVG_FOULS = 24.0
    
    ref_fouls = referee_stats.get('avg_fouls', AVG_FOULS)
    
    # Strictness ratio
    return ref_fouls / AVG_FOULS

def calculate_match_intensity(
    is_derby: bool, 
    importance_score: float, # e.g. 0.0 to 1.0 (cup final = 1.0)
    history_fouls_avg: float
) -> float:
    """
    Calculate a scalar 'Intensity Score' for the match.
    Ranges roughly from 0.8 (friendly) to 1.5 (heated derby).
    """
    intensity = 1.0
    
    if is_derby:
        intensity += 0.2
        
    # Importance impact
    intensity += (importance_score * 0.15)
    
    # Historical heat (if H2H matches usually have high fouls)
    if history_fouls_avg > 28:
        intensity += 0.1
        
    return intensity

def predict_fouls_advanced(
    base_model_fouls: float,
    referee_strictness: float,
    match_intensity: float
) -> int:
    """
    Adjust the machine learning base prediction with context specific factors.
    """
    adjusted = base_model_fouls * referee_strictness * match_intensity
    return int(round(adjusted))

def predict_cards(
    predicted_fouls: int,
    referee_card_rate: float # Cards per foul ratio
) -> Dict[str, Any]:
    """
    Predict yellow/red cards based on foul volume and referee tendency.
    """
    # Heuristic: roughly 1 card every 6-7 fouls is standard, but varies by ref
    if referee_card_rate <= 0:
        referee_card_rate = 0.15 # Default ~1 card per 6.6 fouls
        
    expected_cards = predicted_fouls * referee_card_rate
    
    # Probabilities using Poisson
    from scipy.stats import poisson
    
    prob_over_3_5 = 1 - poisson.cdf(3, expected_cards)
    prob_over_4_5 = 1 - poisson.cdf(4, expected_cards)
    
    return {
        "expected_cards": round(expected_cards, 1),
        "prob_over_3_5": prob_over_3_5,
        "prob_over_4_5": prob_over_4_5
    }
