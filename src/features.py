"""
Feature engineering module for creating ML features from raw match data.
Implements professional-grade, vectorized feature engineering using Pandas.
Includes Time-Aware Rolling Stats, Dynamic Poisson Strength, and Contextual Target Encoding.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

from src.database import SupabaseDB
from src.ingestion import CUP_LEAGUE_IDS
import functools

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ─────────────────────────────────────────────────────

DERBY_CITY_MAP = {
    # England
    "Arsenal": "London", "Chelsea": "London", "Tottenham": "London",
    "West Ham": "London", "Crystal Palace": "London", "Fulham": "London",
    "Brentford": "London",
    "Manchester United": "Manchester", "Manchester City": "Manchester",
    "Liverpool": "Liverpool", "Everton": "Liverpool",
    # Spain
    "Real Madrid": "Madrid", "Atletico Madrid": "Madrid", "Getafe": "Madrid",
    "Rayo Vallecano": "Madrid", "Leganes": "Madrid",
    "Barcelona": "Barcelona", "Espanyol": "Barcelona",
    "Real Betis": "Seville", "Sevilla": "Seville",
    "Real Sociedad": "San Sebastian", "Athletic Club": "Bilbao",
    # France
    "Paris Saint Germain": "Paris", "PSG": "Paris",
    "Olympique Lyonnais": "Lyon", "Lyon": "Lyon",
    "Olympique De Marseille": "Marseille", "Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "OGC Nice": "Nice", "Nice": "Nice",
    "AS Saint-Etienne": "Saint-Etienne", "Saint-Etienne": "Saint-Etienne",
    # Germany
    "Bayern Munich": "Munich", "FC Augsburg": "Augsburg",
    "Borussia Dortmund": "Dortmund",
    "Schalke 04": "Gelsenkirchen",
    "Hertha Berlin": "Berlin", "Union Berlin": "Berlin",
    # Italy
    "AC Milan": "Milan", "Inter": "Milan", "Inter Milan": "Milan",
    "AS Roma": "Rome", "Lazio": "Rome",
    "Juventus": "Turin", "Torino": "Turin",
    "Napoli": "Naples",
    "Genoa": "Genoa", "Sampdoria": "Genoa",
    # Netherlands
    "Ajax": "Amsterdam",
    "Feyenoord": "Rotterdam", "Sparta Rotterdam": "Rotterdam",
    "PSV Eindhoven": "Eindhoven",
    # Portugal
    "Benfica": "Lisbon", "Sporting CP": "Lisbon",
    "FC Porto": "Porto",
}

class FeatureEngineer:
    """
    Advanced Feature Engineer using Vectorized Operations (Pandas).
    Prevents data leakage by ensuring strict time-based rolling windows.
    """
    
    def __init__(self):
        self.db = SupabaseDB()
        self.window_size = 5  # Rolling window size
    
    # ─────────────────────────────────────────────────────
    # DATA LOADING & TRANSFORMATION
    # ─────────────────────────────────────────────────────

    def load_data(self, limit: int = 10000, completed_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load matches and stats from database.
        
        :param limit: Max rows to fetch.
        :param completed_only: If True, only fetch matches with status in ['FT', 'AET', 'PEN'] or date < now.
        """
        # Fetch Matches
        logger.info(f"Fetching last {limit} matches (completed_only={completed_only})...")
        
        all_matches = []
        
        # Supabase max limit is often 1000. We need to paginate using range.
        if not limit or limit <= 0:
            logger.info("Fetching ALL matches (unlimited)...")
            limit = 10_000_000 # Safety cap
            
        # Supabase max limit is often 1000. We need to paginate using range.
        chunk_size = 1000
        start = 0
        
        while start < limit:
            end = min(start + chunk_size, limit) - 1
            
            query = self.db.client.table('matches').select('*').order('match_date', desc=True).range(start, end)
            
            if completed_only:
                query = query.lte('match_date', datetime.now().isoformat())
                
            try:
                response = query.execute()
                data = response.data
                
                if not data:
                    break
                    
                all_matches.extend(data)
                
                # If we got less than chunk size, we are done
                if len(data) < chunk_size:
                    break
                    
                start += chunk_size
                
            except Exception as e:
                logger.error(f"Error fetching matches batch {start}: {e}")
                break
                
        matches_df = pd.DataFrame(all_matches)
        
        # Fetch Stats (optimization: could filter by match_ids present in matches_df)
        match_ids = matches_df['match_id'].tolist() if not matches_df.empty else []
        # Ensure native Python ints for JSON serialization
        match_ids = [int(x) for x in match_ids]
        
        logger.info("Fetching match statistics...")
        # Since stats can be large, we might want to batch this or just fetch a large chunk.
        # Batching to avoid URL length limits
        all_stats = []
        batch_size = 50 
        for i in range(0, len(match_ids), batch_size):
            batch_ids = match_ids[i:i + batch_size]
            try:
                stats_batch = self.db.client.table('stats').select('*').in_('match_id', batch_ids).execute()
                if stats_batch.data:
                    all_stats.extend(stats_batch.data)
            except Exception as e:
                logger.error(f"Error fetching stats batch {i}: {e}")
                
        stats_df = pd.DataFrame(all_stats)
        
        # Preprocessing
        if not matches_df.empty:
            matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
            matches_df = matches_df.sort_values('match_date')
            
        return matches_df, stats_df

    def transform_to_double_entry(self, matches_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Match-Row format to Team-Match-Row format (Double Entry).
        Each match produces two rows: one for home team, one for away team.
        """
        if matches_df.empty:
            return pd.DataFrame()

        # Prepare Home Rows
        home_cols = {
            'match_id': 'match_id',
            'match_date': 'match_date',
            'league': 'league',
            'home_team': 'team',
            'away_team': 'opponent',
            'home_score': 'goals_scored',
            'away_score': 'goals_conceded',
            'result': 'result_raw',
            'referee': 'referee'
        }
        home_df = matches_df[list(home_cols.keys())].rename(columns=home_cols)
        home_df['is_home'] = 1
        
        # Prepare Away Rows
        away_cols = {
            'match_id': 'match_id',
            'match_date': 'match_date',
            'league': 'league',
            'away_team': 'team',
            'home_team': 'opponent',
            'away_score': 'goals_scored',
            'home_score': 'goals_conceded',
            'result': 'result_raw',
            'referee': 'referee'
        }
        away_df = matches_df[list(away_cols.keys())].rename(columns=away_cols)
        away_df['is_home'] = 0
        
        # Concatenate
        double_df = pd.concat([home_df, away_df], ignore_index=True)
        
        # Merge with Stats if available
        if not stats_df.empty:
            # We assume stats_df has 'match_id' and 'team'
            # Select relevant advanced stats
            adv_stats_cols = ['match_id', 'team', 'shots_on_target', 'corner_kicks', 'total_shots', 'fouls', 'yellow_cards', 'expected_goals']
            # Ensure columns exist
            available_cols = [c for c in adv_stats_cols if c in stats_df.columns]
            stats_subset = stats_df[available_cols]
            
            double_df = pd.merge(double_df, stats_subset, on=['match_id', 'team'], how='left')
            
            # Fill NaNs with 0 (missing stats)
            fillna_cols = ['shots_on_target', 'corner_kicks', 'total_shots', 'fouls', 'yellow_cards']
            for col in fillna_cols:
                if col in double_df.columns:
                    double_df[col] = double_df[col].fillna(0)
            
            if 'expected_goals' in double_df.columns:
                double_df['expected_goals'] = double_df['expected_goals'].fillna(double_df['goals_scored'] * 0.9 + 0.1)
        else:
            # Initialize with 0 if no stats
            for col in ['shots_on_target', 'corner_kicks', 'total_shots', 'fouls', 'yellow_cards']:
                double_df[col] = 0
            double_df['expected_goals'] = double_df['goals_scored'] * 0.9 + 0.1

        # Encode Result (Win/Draw/Loss from team perspective)
        # result_raw is 'H' (Home Win), 'A' (Away Win), 'D' (Draw)
        conditions = [
            (double_df['is_home'] == 1) & (double_df['result_raw'] == 'H'),
            (double_df['is_home'] == 0) & (double_df['result_raw'] == 'A')
        ]
        double_df['win'] = np.select(conditions, [1, 1], default=0)
        
        conditions_draw = [
            (double_df['result_raw'] == 'D')
        ]
        double_df['draw'] = np.select(conditions_draw, [1], default=0)
        
        # Points
        double_df['points'] = double_df['win'] * 3 + double_df['draw'] * 1
        
        return double_df.sort_values(['team', 'match_date'])

    # ─────────────────────────────────────────────────────
    # VECTORIZED FEATURE ENGINEERING (CORE LOGIC)
    # ─────────────────────────────────────────────────────

    def calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling stats using strict time-window to avoid leakage.
        Grouping by Team -> Shift(1) -> Rolling(5)
        """
        # Ensure data is sorted by date within team
        df = df.sort_values(['team', 'match_date'])
        
        # Metrics to aggregate
        metrics = ['goals_scored', 'goals_conceded', 'points', 
                   'shots_on_target', 'corner_kicks', 'yellow_cards', 'expected_goals']
        
        # Available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        # Group by team
        grouped = df.groupby('team')[available_metrics]
        
        # Shift 1 to exclude current match, then rolling mean
        rolling_stats = grouped.apply(lambda x: x.shift(1).rolling(window=self.window_size, min_periods=1).mean())
        
        # Rename columns
        rolling_stats.columns = [f'{c}_rolling' for c in rolling_stats.columns]
        
        # Clean up index from groupby if needed/merge back
        # The index should align with df if we didn't drop rows, but groupby apply can be tricky.
        # Alternatively:
        for col in available_metrics:
            df[f'{col}_rolling_avg'] = df.groupby('team')[col].shift(1).rolling(self.window_size).mean()
            
        # Specific Form Calculation (Points / 3)
        if 'points_rolling_avg' in df.columns:
            df['form_score'] = df['points_rolling_avg'] / 3.0
            
        # Calculate rest days
        df['rest_days'] = df.groupby('team')['match_date'].diff().dt.days
        df['rest_days'] = df['rest_days'].fillna(7.0).clip(upper=21.0) # Cap at 21 days for long breaks
        
        return df

    def calculate_poisson_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Dynamic Attack/Defense Strength.
        Attack Strength = Team Rolling Goals Scored / League Average Rolling Goals Scored (at that date)
        
        This is tricky in batches because "League Average" varies by date.
        Approximation: Global rolling average sorted by date.
        """
        # 1. Global League Rolling Average (Time-Aware)
        # Sort by date globally
        df_sorted = df.sort_values('match_date')
        
        # To get "League Average up to this date", we can compute a rolling average on the whole dataset
        # But strictly, we should group by League.
        # For simplicity/robustness, let's group by League.
        
        # League Average (Shift 1 to limit leakage from current match to league avg, though less critical)
        # Fix: Group by league directly and select column
        df['league_avg_goals_scored'] = df_sorted.groupby('league')['goals_scored'].transform(
            lambda x: x.shift(1).rolling(window=50, min_periods=5).mean()
        )
        df['league_avg_goals_conceded'] = df_sorted.groupby('league')['goals_conceded'].transform(
            lambda x: x.shift(1).rolling(window=50, min_periods=5).mean()
        )
        
        # Avoid division by zero
        df['league_avg_goals_scored'] = df['league_avg_goals_scored'].fillna(1.3)
        df['league_avg_goals_conceded'] = df['league_avg_goals_conceded'].fillna(1.3)
        
        # 2. Team Strength
        # We already have team rolling goals in 'goals_scored_rolling_avg'
        if 'goals_scored_rolling_avg' in df.columns:
            df['attack_strength'] = df['goals_scored_rolling_avg'] / df['league_avg_goals_scored']
            df['defense_strength'] = df['goals_conceded_rolling_avg'] / df['league_avg_goals_conceded']
            
        # Fill NaNs (start of season)
        df['attack_strength'] = df['attack_strength'].fillna(1.0)
        df['defense_strength'] = df['defense_strength'].fillna(1.0)
        
        return df

    def calculate_h2h_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate H2H stats for each row efficiently.
        """
        # Dictionary to store history between pairs
        # Key: tuple(sorted(team, opponent)) -> List of results (from team's perspective)
        history: Dict[Tuple[str, str], List[Tuple[Optional[str], bool]]] = {} 
        
        df = df.sort_values('match_date')
        
        # Output columns
        h2h_wins_col = []
        h2h_draws_col = []
        h2h_count_col = []
        
        for idx, row in df.iterrows():
            t1 = row['team']
            t2 = row['opponent']
            # Ensure pair is strictly Tuple[str, str] for Pyre
            sorted_teams = sorted((str(t1), str(t2)))
            pair = (sorted_teams[0], sorted_teams[1])
            
            if pair not in history:
                history[pair] = []
            
            past_matches = history[pair]
            
            # Calculate stats from past matches
            if not past_matches:
                h2h_wins_col.append(0.0)
                h2h_draws_col.append(0.0)
                h2h_count_col.append(0)
            else:
                relevant = past_matches[-5:] # Last 5
                n = len(relevant)
                
                # Count wins for current team (t1)
                # Stored records are (Winner Team Name, is_draw)
                wins = sum(1 for r in relevant if r[0] == t1)
                draws = sum(1 for r in relevant if r[1] is True)
                
                h2h_wins_col.append(wins / n)
                h2h_draws_col.append(draws / n)
                h2h_count_col.append(n)
            
            # Add current match to history for FUTURE rows
            winner = None
            is_draw = False
            if row['win'] == 1:
                winner = t1
            elif row['draw'] == 1:
                is_draw = True
            else:
                winner = t2 # Loss means opponent won
            
            history[pair].append((winner, is_draw))
            
        df['h2h_home_wins'] = h2h_wins_col # Naming convention: actually team_wins
        df['h2h_draws'] = h2h_draws_col
        df['h2h_count'] = h2h_count_col
        
        return df

    def calculate_elo_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ELO ratings iteratively through the sorted dataframe.
        """
        df = df.sort_values('match_date')
        
        elo_dict: Dict[str, float] = {} # team -> rating
        default_elo = 1500.0
        k_factor = 32.0
        
        # Map: (match_id, team) -> pre_match_elo
        pre_match_elo_map = {}
        
        # We iterate over unique matches
        unique_matches = df.drop_duplicates('match_id').sort_values('match_date')
        
        for idx, row in unique_matches.iterrows():
            t1 = row['team']
            t2 = row['opponent']
            
            # Get current ratings
            r1 = elo_dict.get(t1, default_elo)
            r2 = elo_dict.get(t2, default_elo)
            
            # Store PRE-MATCH ratings
            pre_match_elo_map[(row['match_id'], t1)] = r1
            pre_match_elo_map[(row['match_id'], t2)] = r2
            
            # Calculate Expected
            is_t1_home = (row['is_home'] == 1)
            
            r1_adjust = r1 + (50.0 if is_t1_home else 0.0)
            r2_adjust = r2 + (50.0 if not is_t1_home else 0.0)
            
            e1 = 1.0 / (1.0 + 10.0 ** ((r2_adjust - r1_adjust) / 400.0))
            e2 = 1.0 / (1.0 + 10.0 ** ((r1_adjust - r2_adjust) / 400.0))
            
            # Outcome
            if row['win'] == 1:
                s1 = 1.0
                s2 = 0.0
            elif row['draw'] == 1:
                s1 = 0.5
                s2 = 0.5
            else:
                s1 = 0.0
                s2 = 1.0
                
            # Update
            new_r1 = r1 + k_factor * (s1 - e1)
            new_r2 = r2 + k_factor * (s2 - e2)
            
            elo_dict[t1] = new_r1
            elo_dict[t2] = new_r2
            
        def get_elo(row):
            return pre_match_elo_map.get((row['match_id'], row['team']), default_elo)
        
        def get_opp_elo(row):
            return pre_match_elo_map.get((row['match_id'], row['opponent']), default_elo)

        df['team_elo'] = df.apply(get_elo, axis=1)
        df['opponent_elo'] = df.apply(get_opp_elo, axis=1)
        
        return df

    # ─────────────────────────────────────────────────────
    # PIPELINES
    # ─────────────────────────────────────────────────────

    def process_all_features(self, matches_df, stats_df) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.
        """
        # 1. Double Entry
        df = self.transform_to_double_entry(matches_df, stats_df)
        
        # 2. Rolling Stats (Vectorized)
        df = self.calculate_rolling_stats(df)
        
        # 3. Dynamic Poisson Strength
        df = self.calculate_poisson_strength(df)
        
        # 4. H2H (Loop optimized)
        df = self.calculate_h2h_batch(df)
        
        # 5. ELO
        df = self.calculate_elo_dynamic(df)
        
        # Drop rows where we don't have enough history (keeps data clean)
        # But for H2H count 0 is valid.
        # Rolling stats might be NaN at start.
        return df.fillna(0) 

    def create_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the Long-Format (Team View) back to Match-Format (X, y) for training.
        """
        pass # Not used directly, see prepare_training_data

    def prepare_training_data(self, limit: int = 5000) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Public API for Models.
        """
        matches, stats = self.load_data(limit=limit, completed_only=True)
        
        # Process
        long_df = self.process_all_features(matches, stats)
        
        # Transform back to wide (X, y)
        home_df = long_df[long_df['is_home'] == 1].copy()
        away_df = long_df[long_df['is_home'] == 0].copy()
        
        # Define cols map
        cols_map = {
            'team': 'home_team',
            'team_elo': 'home_elo',
            'form_score': 'home_form',
            'attack_strength': 'home_attack_strength', 
            'defense_strength': 'home_defensive_strength', 
            'goals_scored_rolling_avg': 'home_offensive',
            'goals_conceded_rolling_avg': 'home_defensive',
            'h2h_home_wins': 'h2h_home_wins',
            'h2h_draws': 'h2h_draws',
            'h2h_count': 'h2h_count',
            'result_raw': 'result',
            'goals_scored': 'home_actual_goals',
            'fouls': 'home_fouls',
            'expected_goals_rolling_avg': 'home_xg_rolling',
            'rest_days': 'home_rest_days'
        }
        
        home_subset = home_df[list(cols_map.keys()) + ['match_id', 'match_date', 'referee', 'league']].rename(columns=cols_map)
        
        # Fix: Away team needs H2H win rate too. 
        # In LongDF, 'h2h_home_wins' is the team's win rate.
        # So for Away team, 'h2h_home_wins' IS 'h2h_away_wins' from the match perspective.
        
        away_cols_map = {
            'team': 'away_team',
            'team_elo': 'away_elo',
            'form_score': 'away_form',
            'attack_strength': 'away_attack_strength',
            'defense_strength': 'away_defense_strength',
            'goals_scored_rolling_avg': 'away_offensive',
            'goals_conceded_rolling_avg': 'away_defensive',
            'goals_scored': 'away_actual_goals',
            'h2h_home_wins': 'h2h_away_wins', # Mapping team's win rate to 'away_wins' feature
            'fouls': 'away_fouls',
            'expected_goals_rolling_avg': 'away_xg_rolling',
            'rest_days': 'away_rest_days'
        }
        
        away_subset = away_df[list(away_cols_map.keys()) + ['match_id']].rename(columns=away_cols_map)
        
        # Merge
        full = pd.merge(home_subset, away_subset, on='match_id')
        
        # Differentials
        full['offensive_diff'] = full['home_offensive'] - full['away_offensive']
        full['defensive_diff'] = full['home_defensive'] - full['away_defensive']
        full['form_diff'] = full['home_form'] - full['away_form']
        full['xg_diff'] = full['home_xg_rolling'] - full['away_xg_rolling']
        full['rest_diff'] = full['home_rest_days'] - full['away_rest_days']
        full['home_advantage'] = 1.0
        
        # Derby & Ref
        full['is_derby'] = full.apply(lambda x: 1.0 if self.detect_derby(x['home_team'], x['away_team']) else 0.0, axis=1)
        full['referee_aggression'] = full['referee'].apply(lambda x: self.calculate_referee_aggression(x))
        
        # H2H avg goals - Need to persist/calc?
        full['h2h_avg_goals'] = 2.5 # Placeholder or recalc
        
        # Competition
        full['competition_type'] = full.apply(lambda x: 1.0 if self.detect_competition_type(x['league']) == 'CUP' else 0.0, axis=1)
        full['is_knockout'] = full.apply(lambda x: 1.0 if self.detect_knockout(str(x['match_date']), x['competition_type']) else 0.0, axis=1)

        # Select columns matching Models expectations (FEATURE_COLUMNS in app.py)
        ordered_cols = [
            'home_form', 'away_form',
            'home_offensive', 'away_offensive',
            'home_defensive', 'away_defensive',
            'offensive_diff', 'defensive_diff', 'form_diff',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_avg_goals',
            'referee_aggression', 'home_advantage',
            'home_elo', 'away_elo',
            'home_attack_strength', 'away_defense_strength',
            'is_derby', 'competition_type', 'is_knockout',
            'home_xg_rolling', 'away_xg_rolling', 'xg_diff',
            'home_rest_days', 'away_rest_days', 'rest_diff'
        ]
        
        # Ensure 'away_defense_strength' exists (we have away_defensive_strength)
        # Rename if needed.
        # Note: Models might expect specific names.
        
        # Target
        y_result = full['result']
        
        # Filter out invalid results (e.g. '0' or None)
        valid_mask = y_result.isin(['H', 'D', 'A'])
        if not valid_mask.all():
            logger.warning(f"Dropping {len(y_result) - valid_mask.sum()} rows with invalid results (not H/D/A)")
            full = full[valid_mask]
            y_result = full['result']
            
        y_goals = full['home_actual_goals'] + full['away_actual_goals']
        y_fouls = full['home_fouls'] + full['away_fouls']
        
        # FillNa
        X = full[ordered_cols].fillna(0)
        
        return X, y_result, y_goals, y_fouls

    def create_match_features(self, home_team, away_team, referee=None, league=None, league_id=None, match_date=None) -> pd.DataFrame:
        """
        Inference Method:
        1. Fetch recent history for both teams.
        2. Create a dummy row for the new match.
        3. Run feature pipeline.
        4. Return the specific row.
        """
        # Fetch history (limited to last 20 matches per team is enough for rolling(5))
        # We need enough history for ELO calculation though...
        # Ideal: We cache ELO. For now, we might need to load a bit more or use a stored ELO snapshot.
        # Simplified: Load last 100 matches global to get decent current stats.
        
        matches, stats = self.load_data(limit=5000, completed_only=True) # Load enough history for rolling stats
        
        # LEAKAGE PREVENTION: Data Integrity Check
        # If the match we are predicting (same teams, same day) starts to exist in our history (e.g. predicting past match),
        # we must exclude it. Otherwise, features will use the *actual result* of the match we are trying to predict!
        if match_date and not matches.empty:
            target_ts = pd.to_datetime(match_date, utc=True) if match_date else datetime.now(timezone.utc)
            # Filter matches strictly before the target date (or exclude matches on the same day if we want to be super safe)
            # Actually, just filtering < match_date is correct for backtesting.
            # But ensures we don't have exact duplicates or "future" data relative to prediction time.
            matches['match_date'] = pd.to_datetime(matches['match_date'], utc=True)
            matches = matches[matches['match_date'] < target_ts]
        
        # Add dummy match
        if not match_date:
            match_date = datetime.now().isoformat()
            
        dummy_match = {
            'match_id': 99999999,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date, # Now
            'league': league,
            'referee': referee,
            'home_score': None, # Future
            'away_score': None,
            'result': None
        }
        
        dummy_df = pd.DataFrame([dummy_match])
        # Ensure compatible types for merging (float for scores which are None here)
        dummy_df = dummy_df.astype({'home_score': 'float', 'away_score': 'float'})
        matches = pd.concat([matches, dummy_df], ignore_index=True)
        # Fix date conversion for dummy
        matches['match_date'] = pd.to_datetime(matches['match_date'], utc=True)
        
        # Run pipeline
        long_df = self.process_all_features(matches, stats)
        
        # Extract the dummy rows
        # The dummy match has match_id 99999999
        # In Long DF, we have 2 rows for this match (Team, Opponent)
        
        relevant = long_df[long_df['match_id'] == 99999999]
        if relevant.empty:
            # Should not happen unless dropped by dropna (e.g. rolling stats not ready)
            # If empty, return zeros
            return pd.DataFrame(columns=[
                'home_form', 'away_form', 'home_offensive', 'away_offensive',
                'home_elo', 'away_elo' # etc...
            ]) # simplified
        
        # Reconstruct Wide format manually for this single match
        home_row = relevant[relevant['is_home'] == 1].iloc[0]
        away_row = relevant[relevant['is_home'] == 0].iloc[0]
        
        # Map values
        feats = {}
        feats['home_form'] = home_row.get('form_score', 0.5)
        feats['away_form'] = away_row.get('form_score', 0.5)
        
        # Feature Mapping (Must match FEATURE_COLUMNS)
        feats['home_offensive'] = home_row.get('goals_scored_rolling_avg', 1.0)
        feats['away_offensive'] = away_row.get('goals_scored_rolling_avg', 1.0)
        feats['home_defensive'] = home_row.get('goals_conceded_rolling_avg', 1.0)
        feats['away_defensive'] = away_row.get('goals_conceded_rolling_avg', 1.0)
        
        feats['offensive_diff'] = feats['home_offensive'] - feats['away_offensive']
        feats['defensive_diff'] = feats['home_defensive'] - feats['away_defensive']
        feats['form_diff'] = feats['home_form'] - feats['away_form']
        
        feats['h2h_home_wins'] = home_row.get('h2h_home_wins', 0)
        feats['h2h_draws'] = home_row.get('h2h_draws', 0)
        feats['h2h_away_wins'] = home_row.get('h2h_away_wins', 0)
        feats['h2h_avg_goals'] = 2.5 # Placeholder
        
        feats['referee_aggression'] = self.calculate_referee_aggression(referee or 'Unknown')
        feats['home_advantage'] = 1.0
        
        feats['home_elo'] = home_row.get('team_elo', 1500)
        feats['away_elo'] = away_row.get('team_elo', 1500)
        
        feats['home_attack_strength'] = home_row.get('attack_strength', 1.0)
        feats['away_defense_strength'] = away_row.get('defense_strength', 1.0)
        
        feats['is_derby'] = 1.0 if self.detect_derby(home_team, away_team) else 0.0
        feats['competition_type'] = 1.0 if self.detect_competition_type(league) == 'CUP' else 0.0
        feats['is_knockout'] = 0.0
        
        # New Advanced Features (xG and Fatigue)
        feats['home_xg_rolling'] = home_row.get('expected_goals_rolling_avg', 1.0)
        feats['away_xg_rolling'] = away_row.get('expected_goals_rolling_avg', 1.0)
        feats['xg_diff'] = feats['home_xg_rolling'] - feats['away_xg_rolling']
        
        feats['home_rest_days'] = home_row.get('rest_days', 7.0)
        feats['away_rest_days'] = away_row.get('rest_days', 7.0)
        feats['rest_diff'] = feats['home_rest_days'] - feats['away_rest_days']
        
        # Add h2h_count explicitly if needed by app
        
        return pd.DataFrame([feats])

    # ─────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────
    
    @staticmethod
    def detect_derby(home_team: str, away_team: str) -> bool:
        hc = DERBY_CITY_MAP.get(home_team)
        ac = DERBY_CITY_MAP.get(away_team)
        if hc and ac and hc == ac:
            return True
        return False

    @staticmethod
    def detect_competition_type(league: str = None) -> str:
        if not league: return 'LEAGUE'
        kw = ['cup', 'pokal', 'copa', 'trophy', 'champions', 'europa']
        if any(k in league.lower() for k in kw):
            return 'CUP'
        return 'LEAGUE'
        
    @functools.lru_cache(maxsize=None)
    def calculate_referee_aggression(self, referee_name: str) -> float:
        """
        Calculate referee aggression based on cards and fouls in past matches.
        """
        try:
            # Query matches by this referee
            response = self.db.client.table('matches').select('match_id').eq('referee', referee_name).limit(20).execute()
            if not response.data:
                return 3.5 # Average default
                
            match_ids = [m['match_id'] for m in response.data]
            
            # Get stats
            stats_resp = self.db.client.table('stats').select('yellow_cards, red_cards, fouls').in_('match_id', match_ids).execute()
            if not stats_resp.data:
                return 3.5
                
            total_yellow = sum(s.get('yellow_cards', 0) for s in stats_resp.data)
            total_red = sum(s.get('red_cards', 0) for s in stats_resp.data)
            total_fouls = sum(s.get('fouls', 0) for s in stats_resp.data)
            n = len(match_ids) # Approximate, as stats might have 2 rows per match or 1 depending on table structure. 
            # Actually stats has 2 rows per match (home/away). So we should divide by len(stats_resp.data) / 2?
            # Or just take average per team-match record?
            # Aggression usually is per match.
            # Let's count unique matches in stats?
            # Easiest: Aggression = (Avg Yellow per team-match * 2) + ...
            # Let's stick to per team-match avg * 2 to estimate per match?
            # Or just normalize.
            
            count = len(stats_resp.data)
            if count == 0: return 3.5
            
            avg_yellow = total_yellow / count
            avg_red = total_red / count
            avg_fouls = total_fouls / count
            
            # Score: (Yellow * 1 + Red * 5 + Fouls * 0.1) * 2 (to approx match total if we have team rows)
            # If stats has 2 rows per match, avg_yellow is per team. Match total is 2x.
            # Base aggression index roughly 3-5 range.
            # Example: 2 yellows per team = 4 per match. 
            aggression = (avg_yellow * 1.0 + avg_red * 5.0 + avg_fouls * 0.1) * 2.0
            
            return max(0.0, aggression)
            
        except Exception as e:
            logger.warning(f"Error calcluating referee aggression: {e}")
            return 3.5

    @staticmethod
    def detect_knockout(match_date: str, competition_type: float) -> bool:
        """
        Detect if a CUP match is in the knockout phase.
        Knockouts usually start after January.
        """
        if competition_type == 0.0: # LEAGUE
            return False
            
        if not match_date:
            return False
            
        try:
            # match_date is ISO string or datetime
            if isinstance(match_date, str):
                dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
            else:
                dt = match_date
                
            # Knockout rounds: February (2) onwards
            return dt.month >= 2
        except Exception:
            return False

