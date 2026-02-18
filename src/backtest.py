
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from src.features import FeatureEngineer
from src.models import MatchPredictor
from src.strategy import detect_value_bet, calculate_kelly_stake

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(season_start_date: str = '2024-08-01', initial_bankroll: float = 1000.0):
    logger.info("🚀 Starting Walk-Forward Backtest...")
    
    # 1. Load ALL Data
    fe = FeatureEngineer()
    matches, stats = fe.load_data(limit=10000)
    logger.info(f"Loaded {len(matches)} matches.")
    
    # Convert matches to DF for odds and dates
    matches_df = pd.DataFrame(matches)
    matches_df['match_date'] = pd.to_datetime(matches_df['match_date'], utc=True)
    
    # 2. Feature Engineering (Batch)
    logger.info("Generating features (this may take a moment)...")
    long_df = fe.process_all_features(matches, stats)
    
    # 3. Pivot to Wide Format (One row per match)
    # Filter for Home rows to be the base
    home_df = long_df[long_df['is_home'] == 1].copy()
    away_df = long_df[long_df['is_home'] == 0].copy()
    
    # Select columns to merge from away
    # We want features from away team, typically renamed to 'away_Feature'
    # Exclude metadata that is common or specific to home
    exclude_cols = ['match_id', 'match_date', 'season', 'league_id', 'is_home', 'home_team', 'away_team', 'home_score', 'away_score', 'result', 'win', 'draw', 'loss']
    feature_cols = [c for c in away_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(away_df[c])]
    
    away_subset = away_df[['match_id'] + feature_cols].rename(columns={c: f"away_{c}" for c in feature_cols})
    
    # Merge Home and Away features
    full_df = pd.merge(home_df, away_subset, on='match_id', how='inner')
    
    # Merge with matches_df to ensure we have Odds and clean Metadata
    # Add scores to compute result if missing
    # Add status to filter completed matches
    meta_cols = ['match_id', 'odds_home', 'odds_draw', 'odds_away', 'home_score', 'away_score']
    if 'status' in matches_df.columns:
        meta_cols.append('status')
    if 'result' in matches_df.columns:
        meta_cols.append('result')
        
    full_df = pd.merge(full_df, matches_df[meta_cols], on='match_id', how='inner')
    
    # Filter for Completed Matches Only
    if 'status' in full_df.columns:
        # Check specific status like 'FT', 'AET', 'PEN' or just exclude 'NS', 'PST'
        # Safest: status in ['FT', 'AET', 'PEN']
        full_df = full_df[full_df['status'].isin(['FT', 'AET', 'PEN'])]
    
    # Compute Reult if not present
    
    # Compute Reult if not present
    if 'result' not in full_df.columns:
        conditions = [
            (full_df['home_score'] > full_df['away_score']),
            (full_df['home_score'] < full_df['away_score'])
        ]
        choices = ['H', 'A']
        full_df['result'] = np.select(conditions, choices, default='D')
    
    # Ensure date is datetime
    full_df['match_date'] = pd.to_datetime(full_df['match_date'], utc=True)
    full_df = full_df.sort_values('match_date')
    
    # Features list for model (exclude targets/meta)
    # We assume 'result' is the target. 'home_score', 'away_score' are for regressors.
    drop_cols = [
        'match_id', 'match_date', 'season', 'league_id', 
        'is_home', 'team', 'opponent', 'home_team', 'away_team',
        'home_score', 'away_score', 'result', 'win', 'draw', 'loss',
        'odds_home', 'odds_draw', 'odds_away', 'week'
    ]
    # Filter for numeric columns only + explicitly exclude bad ones
    X_cols = [c for c in full_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(full_df[c])]
    
    logger.info(f"Feature columns ({len(X_cols)}): {X_cols[:5]} ...")
    
    # Define Weeks globally on full_df
    if full_df['match_date'].dt.tz is None:
        full_df['match_date'] = full_df['match_date'].dt.tz_localize('UTC')
    else:
        full_df['match_date'] = full_df['match_date'].dt.tz_convert('UTC')
        
    full_df['week'] = full_df['match_date'].dt.to_period('W-MON').dt.start_time
    # Force UTC on week
    if full_df['week'].dt.tz is None:
         full_df['week'] = full_df['week'].dt.tz_localize('UTC')
    else:
         full_df['week'] = full_df['week'].dt.tz_convert('UTC')

    # 4. Filter Simulation Period
    start_dt = pd.to_datetime(season_start_date, utc=True)
    sim_df = full_df[full_df['match_date'] >= start_dt].copy()
    
    if sim_df.empty:
        logger.error("No matches found for simulation period.")
        return

    weeks = sorted(sim_df['week'].unique())
    
    logger.info(f"Simulating {len(weeks)} weeks with {len(sim_df)} matches.")
    
    # 5. Walk-Forward Loop
    bankroll = initial_bankroll
    current_capital = bankroll
    capital_history = [current_capital]
    dates = [start_dt]
    
    predictor = MatchPredictor()
    
    total_bets = 0
    wins = 0
    total_staked = 0.0
    
    for current_week in weeks:
        week_str = current_week.strftime('%Y-%m-%d')
        
        # A. Split Data
        train_data = full_df[full_df['match_date'] < current_week]
        test_data = full_df[full_df['week'] == current_week]
        
        if train_data.empty or test_data.empty:
            continue
            
        # Refine Training Data
        X_train = train_data[X_cols]
        y_train = train_data['result'].astype(str) # Encode target
        
        if len(X_train) < 50: # Minimum samples check
            continue
            
        # B. Retrain Model (Result Classifier)
        # We assume lightweight training or it might be slow. 
        # For a strict backtest, we retrain.
        # To speed up, we could retrain every N weeks, but user asked for "At each week".
        # We suppress logs for cleaner output
        logging.getLogger('src.models').setLevel(logging.WARNING)
        predictor.train_result_classifier(X_train, y_train)
        
        # C. Predict
        X_test = test_data[X_cols]
        try:
            probs = predictor.result_pipeline.predict_proba(X_test)
            classes = predictor.label_encoder.classes_ # ['A', 'D', 'H'] usually
        except Exception as e:
            logger.error(f"Prediction failed for week {week_str}: {e}")
            continue
            
        # D. Simulate Betting
        for idx, (array_idx, row) in enumerate(test_data.iterrows()):
            # Get model probabilities
            prob_dict = dict(zip(classes, probs[idx]))
            
            p_home = prob_dict.get('H', 0.0)
            p_draw = prob_dict.get('D', 0.0)
            p_away = prob_dict.get('A', 0.0)
            
            # Get valid odds
            odds_h = row.get('odds_home')
            odds_d = row.get('odds_draw')
            odds_a = row.get('odds_away')
            
            if not (odds_h and odds_d and odds_a):
                continue
                
            # Detect Value
            # Check all 3 outcomes
            outcomes = [
                ('H', p_home, odds_h),
                ('D', p_draw, odds_d),
                ('A', p_away, odds_a)
            ]
            
            best_bet = None
            max_ev = 0.0
            
            for label, prob, odd in outcomes:
                ev = detect_value_bet(prob, odd)
                if ev > max_ev:
                    max_ev = ev
                    best_bet = (label, prob, odd)
            
            # Place Bet if Value found
            if max_ev > 0.05: # 5% Edge
                label, prob, odd = best_bet
                stake = calculate_kelly_stake(prob, odd, current_capital, fractional=0.25)
                
                if stake > 0:
                    total_bets += 1
                    total_staked += stake
                    current_capital -= stake
                    
                    # Verify Result
                    actual_result = row['result'] # 'H', 'D', 'A'
                    
                    won = (actual_result == label)
                    profit = 0.0
                    
                    if won:
                        wins += 1
                        revenue = stake * odd
                        current_capital += revenue
                        profit = revenue - stake
                    else:
                        profit = -stake
                        
                    # logger.info(f"Bet {label} @ {odd:.2f} | Stake: {stake:.1f}€ | Res: {actual_result} | Won: {won}")
        
        # End of Week Update
        capital_history.append(current_capital)
        dates.append(current_week)
        logger.info(f"Week {week_str} done. Bankroll: {current_capital:.2f}€")
        
    # 6. Report
    net_profit = current_capital - initial_bankroll
    roi = (net_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0
    
    # Calculate Max Drawdown
    peak = initial_bankroll
    max_dd = 0.0
    for val in capital_history:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
            
    print("\n" + "="*40)
    print(f"BACKTEST REPORT (2024-2025)")
    print("="*40)
    print(f"Initial Bankroll : {initial_bankroll:.2f}€")
    print(f"Final Bankroll   : {current_capital:.2f}€")
    print(f"Net Profit       : {net_profit:+.2f}€")
    print(f"ROI (Yield)      : {roi:+.2f}%")
    print(f"Total Bets       : {total_bets}")
    print(f"Win Rate         : {win_rate:.1f}%")
    print(f"Max Drawdown     : {max_dd*100:.2f}%")
    print("="*40)
    
    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, capital_history, marker='o', linestyle='-')
    plt.title('Bankroll Evolution (Kelly Strategy)')
    plt.xlabel('Date')
    plt.ylabel('Capital (€)')
    plt.grid(True)
    plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial')
    plt.legend()
    plt.savefig('backtest_results.png')
    logger.info("Graph saved to backtest_results.png")

if __name__ == "__main__":
    run_backtest()
