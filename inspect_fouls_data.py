
import pandas as pd
from src.features import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)

def inspect():
    fe = FeatureEngineer()
    print("Loading data...")
    # Load a decent amount to get a representative sample
    X, y_result, y_goals, y_fouls = fe.prepare_training_data(limit=2000)
    
    print(f"Total samples: {len(y_fouls)}")
    print(f"Total samples: {len(y_fouls)}")
    # Explicitly verify it's a series and handle boolean conversion if needed
    zero_fouls = (y_fouls == 0)
    print(f"Zero fouls count: {zero_fouls.sum()}")
    
    # Filter using boolean indexing
    non_zero = y_fouls[y_fouls > 0]
    print(f"Non-zero fouls mean: {non_zero.mean()}")
    print(f"Non-zero fouls min: {non_zero.min()}")
    print(f"Non-zero fouls max: {non_zero.max()}")
    
    if zero_fouls.sum() > 0:
        print("\nWARNING: Found matches with 0 fouls. These are likely missing stats.")
        print("Recommendation: Filter these out before training foul regressor.")
    else:
        print("\nData looks clean (no 0 fouls).")

if __name__ == "__main__":
    inspect()
