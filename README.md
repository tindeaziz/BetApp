# ⚽ Football Match Prediction Application

A production-ready Python application for predicting football match outcomes using Machine Learning and Big Data analytics.

## 🎯 Features

- **Data Ingestion**: Fetch historical match data from API-Football
- **Feature Engineering**: Calculate team form, offensive/defensive strength, referee aggression, and H2H statistics
- **ML Predictions**:
  - **1N2 Classifier**: Predict match result (Home Win / Draw / Away Win)
  - **Over/Under Regressor**: Predict total goals
  - **Fouls Regressor**: Predict total fouls
- **Database**: Supabase (PostgreSQL) for scalable data storage
- **CLI Interface**: Easy-to-use command-line interface

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Database**: Supabase (PostgreSQL)
- **ML Framework**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **API**: API-Football integration

## 📁 Project Structure

```
BetApp/
├── src/
│   ├── __init__.py
│   ├── database.py      # Supabase connection & CRUD operations
│   ├── ingestion.py     # API data fetching & cleaning
│   ├── features.py      # Feature engineering
│   └── models.py        # ML models (XGBoost)
├── models/              # Saved model files (auto-created)
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
├── .env                 # Environment variables (API keys)
└── .gitignore
```

## 🚀 Installation

### 1. Clone or navigate to the project directory

```bash
cd BetApp
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit `.env` file with your credentials:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
API_FOOTBALL_KEY=your_api_football_key
```

**Get your keys:**
- Supabase: [https://supabase.com](https://supabase.com) (free tier available)
- API-Football: [https://www.api-football.com](https://www.api-football.com) (free tier: 100 requests/day)

### 5. Initialize database

Run the SQL schema in your Supabase Dashboard SQL Editor (see Database Setup below).

## 🗄️ Database Setup

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the initialization command to see the SQL schema:

```bash
python main.py init
```

4. Copy the SQL output and execute it in Supabase SQL Editor to create the tables:
   - `matches` - Match information
   - `stats` - Team statistics per match
   - `predictions` - AI predictions

## 📖 Usage

### Initialize Database

```bash
python main.py init
```

### Ingest Historical Data

```bash
# Ingest Premier League 2023 season
python main.py ingest --league 39 --season 2023

# Ingest with date range
python main.py ingest --league 39 --season 2023 --from 2023-08-01 --to 2023-12-31
```

**Common League IDs:**
- Premier League: 39
- La Liga: 140
- Bundesliga: 78
- Serie A: 135
- Ligue 1: 61

### Train ML Models

```bash
# Train with default 1000 matches
python main.py train

# Train with custom data limit
python main.py train --limit 500
```

This will create three models in the `models/` directory:
- `result_classifier.joblib` - 1N2 prediction
- `goals_regressor.joblib` - Over/Under prediction
- `fouls_regressor.joblib` - Fouls prediction

### Predict Match Outcome

```bash
# Basic prediction
python main.py predict --home "Manchester United" --away "Liverpool"

# With referee information
python main.py predict --home "Manchester United" --away "Liverpool" --referee "Michael Oliver"
```

**Output example:**
```
==================================================
MATCH PREDICTION
==================================================
Match: Manchester United vs Liverpool

Result Probabilities:
  Home Win: 45.23%
  Draw: 28.45%
  Away Win: 26.32%

Predicted Result: H
Confidence: 45.23%

Expected Total Goals: 2.73
Expected Total Fouls: 24
==================================================
```

## 🧪 Workflow Example

```bash
# 1. Initialize database
python main.py init

# 2. Ingest data
python main.py ingest --league 39 --season 2023

# 3. Train models
python main.py train --limit 1000

# 4. Make predictions
python main.py predict --home "Arsenal" --away "Chelsea"
```

## 🔍 Features Explained

### Calculated Features

1. **Team Form**: Win/draw/loss ratio over last 5 matches
2. **Offensive Strength**: Shots on target, total shots, xG
3. **Defensive Strength**: Goals conceded, blocks, saves
4. **Referee Aggression**: Average cards and fouls per match
5. **Head-to-Head**: Historical results between teams
6. **Home Advantage**: Constant factor for home team

### Model Architecture

- **Result Classifier**: XGBoost multi-class classifier (3 classes: H/D/A)
- **Goals Regressor**: XGBoost regressor for continuous goal prediction
- **Fouls Regressor**: XGBoost regressor for fouls prediction

## 📊 Model Performance

After training, you'll see metrics like:

```
Result Classifier (1N2):
  Accuracy: 0.5234
  CV Score: 0.5123 (+/- 0.0234)

Goals Regressor (Over/Under):
  RMSE: 1.2345
  R²: 0.3456

Fouls Regressor:
  RMSE: 4.5678
  R²: 0.2345
```

## 🔧 Troubleshooting

### Models not found error
```bash
# Make sure to train models first
python main.py train
```

### Database connection error
- Verify your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check that tables are created in Supabase dashboard

### API rate limit exceeded
- API-Football free tier: 100 requests/day
- Consider upgrading or using cached data

## 🚀 Next Steps

1. **Improve Features**: Add player injuries, weather, venue statistics
2. **Hyperparameter Tuning**: Optimize XGBoost parameters
3. **Ensemble Models**: Combine multiple algorithms
4. **Real-time Predictions**: Build REST API with FastAPI
5. **Web Dashboard**: Create visualization interface
6. **Backtesting**: Validate predictions against historical results

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork and improve this project!

## ⚠️ Disclaimer

This application is for educational and research purposes only. Sports betting involves risk. Always gamble responsibly.
