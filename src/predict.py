"""
🎯 Script Interactif de Pronostic Football
==========================================

Charge les modèles XGBoost entraînés, récupère la forme actuelle
des équipes depuis Supabase, et fournit des prédictions détaillées
avec conseils de paris.

Usage:
    python -m src.predict
    ou
    python src/predict.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SupabaseDB, get_team_matches, get_team_stats
from src.features import FeatureEngineer

# Charger les variables d'environnement
load_dotenv()


# ─────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/")

# Ordre EXACT des features tel qu'utilisé lors de l'entraînement
FEATURE_COLUMNS = [
    'home_form',
    'away_form',
    'home_offensive',
    'away_offensive',
    'home_defensive',
    'away_defensive',
    'offensive_diff',
    'defensive_diff',
    'form_diff',
    'h2h_home_wins',
    'h2h_draws',
    'h2h_away_wins',
    'h2h_avg_goals',
    'referee_aggression',
    'home_advantage',
    # Advanced features
    'home_elo',
    'away_elo',
    'home_attack_strength',
    'away_defense_strength',
    'is_derby',
    # Competition features
    'competition_type',
    'is_knockout'
]


# ─────────────────────────────────────────────────────────
# CHARGEMENT DES CERVEAUX (MODÈLES)
# ─────────────────────────────────────────────────────────

def load_brains() -> dict:
    """
    Charge les 3 modèles XGBoost et le LabelEncoder depuis le disque.
    
    Returns:
        Dictionnaire contenant les modèles et l'encodeur
        
    Raises:
        FileNotFoundError: Si un fichier modèle est manquant
    """
    print("\n🧠 Chargement des cerveaux IA...")
    
    models = {}
    
    files = {
        'result_classifier': 'result_classifier.joblib',
        'goals_regressor': 'goals_regressor.joblib',
        'fouls_regressor': 'fouls_regressor.joblib',
        'label_encoder': 'label_encoder.joblib'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(MODEL_PATH, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ Modèle manquant : {filepath}\n"
                f"   Lancez d'abord : python main.py train"
            )
        models[key] = joblib.load(filepath)
        print(f"   ✅ {filename} chargé")
    
    print("🧠 Tous les cerveaux sont prêts !\n")
    return models


# ─────────────────────────────────────────────────────────
# RÉCUPÉRATION DE LA FORME ACTUELLE
# ─────────────────────────────────────────────────────────

def get_team_recent_form(team_name: str, fe: FeatureEngineer) -> dict:
    """
    Récupère la forme actuelle d'une équipe basée sur ses 5 derniers matchs.
    Recalcule les stats glissantes exactement comme lors de l'entraînement.
    
    Args:
        team_name: Nom de l'équipe
        fe: Instance de FeatureEngineer
        
    Returns:
        Dictionnaire avec les métriques de forme
    """
    form = fe.calculate_team_form(team_name, n_matches=5)
    offensive = fe.calculate_offensive_strength(team_name, n_matches=10)
    defensive = fe.calculate_defensive_strength(team_name, n_matches=10)
    
    return {
        'form': form,
        'offensive': offensive,
        'defensive': defensive
    }


def display_team_form(team_name: str, stats: dict):
    """Affiche les stats de forme d'une équipe de manière visuelle."""
    form_pct = stats['form'] * 100
    
    # Barre de forme visuelle
    filled = int(form_pct / 10)
    bar = "█" * filled + "░" * (10 - filled)
    
    print(f"   📊 Forme récente :  [{bar}] {form_pct:.0f}%")
    print(f"   ⚔️  Force offensive : {stats['offensive']:.2f}")
    print(f"   🛡️  Force défensive : {stats['defensive']:.2f}")


# ─────────────────────────────────────────────────────────
# CONSTRUCTION DES FEATURES
# ─────────────────────────────────────────────────────────

def build_match_features(
    home_team: str,
    away_team: str,
    fe: FeatureEngineer,
    referee: str = None
) -> pd.DataFrame:
    """
    Construit le DataFrame de features pour la prédiction.
    Les colonnes sont dans l'EXACT même ordre que l'entraînement.
    
    Args:
        home_team: Équipe domicile
        away_team: Équipe extérieur
        fe: Instance de FeatureEngineer
        referee: Arbitre (optionnel)
        
    Returns:
        DataFrame avec les features ordonnées
    """
    # Utiliser le même FeatureEngineer que lors de l'entraînement
    features_df = fe.create_match_features(home_team, away_team, referee)
    
    # S'assurer que les colonnes sont dans le bon ordre
    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            features_df[col] = 0.0
    
    # Réordonner les colonnes EXACTEMENT comme l'entraînement
    features_df = features_df[FEATURE_COLUMNS]
    
    return features_df


# ─────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    models: dict,
    fe: FeatureEngineer,
    referee: str = None
) -> dict:
    """
    Effectue la prédiction complète d'un match.
    
    Args:
        home_team: Équipe domicile
        away_team: Équipe extérieur
        models: Dictionnaire des modèles chargés
        fe: FeatureEngineer avec connexion DB
        referee: Arbitre (optionnel)
        
    Returns:
        Dictionnaire avec toutes les prédictions
    """
    # Construire les features
    features = build_match_features(home_team, away_team, fe, referee)
    
    # 1. Prédiction du résultat (1N2)
    result_proba = models['result_classifier'].predict_proba(features)[0]
    result_classes = models['label_encoder'].classes_
    prob_dict = dict(zip(result_classes, result_proba))
    
    predicted_idx = np.argmax(result_proba)
    predicted_result = result_classes[predicted_idx]
    confidence = float(result_proba[predicted_idx])
    
    # 2. Prédiction des buts
    predicted_goals = float(models['goals_regressor'].predict(features)[0])
    predicted_goals = max(0, predicted_goals)  # Pas de buts négatifs
    
    # 3. Prédiction des fautes
    predicted_fouls = int(models['fouls_regressor'].predict(features)[0])
    predicted_fouls = max(0, predicted_fouls)  # Pas de fautes négatives
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'prob_home': float(prob_dict.get('H', 0.0)),
        'prob_draw': float(prob_dict.get('D', 0.0)),
        'prob_away': float(prob_dict.get('A', 0.0)),
        'predicted_result': predicted_result,
        'confidence': confidence,
        'predicted_goals': predicted_goals,
        'predicted_fouls': predicted_fouls,
        'over_2_5': predicted_goals > 2.5,
        'timestamp': datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────────
# AFFICHAGE DES RÉSULTATS
# ─────────────────────────────────────────────────────────

def display_prediction(pred: dict):
    """
    Affiche les résultats de prédiction de manière esthétique.
    
    Args:
        pred: Dictionnaire de prédiction
    """
    home = pred['home_team']
    away = pred['away_team']
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              🎯 PRONOSTIC IA FOOTBALL                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  🏟️  {home} vs {away}")
    print(f"║  📅  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("╠══════════════════════════════════════════════════════════╣")
    
    # Probabilités avec barres visuelles
    print("║")
    print("║  📊 PROBABILITÉS DE RÉSULTAT :")
    print("║")
    
    # Home win
    home_bar = "█" * int(pred['prob_home'] * 30)
    print(f"║  🏠 {home:20s}  {pred['prob_home']*100:5.1f}% │{home_bar}")
    
    # Draw
    draw_bar = "█" * int(pred['prob_draw'] * 30)
    print(f"║  🤝 Match Nul            {pred['prob_draw']*100:5.1f}% │{draw_bar}")
    
    # Away win
    away_bar = "█" * int(pred['prob_away'] * 30)
    print(f"║  🏃 {away:20s}  {pred['prob_away']*100:5.1f}% │{away_bar}")
    
    print("║")
    print("╠══════════════════════════════════════════════════════════╣")
    
    # Stats de match
    print("║")
    print(f"║  ⚽ Buts attendus    : {pred['predicted_goals']:.1f}  ", end="")
    if pred['over_2_5']:
        print("(OVER 2.5 ✅)")
    else:
        print("(UNDER 2.5 ⬇️)")
    
    print(f"║  🟨 Fautes attendues : {pred['predicted_fouls']}")
    print(f"║  🎯 Confiance IA     : {pred['confidence']*100:.1f}%")
    print("║")
    
    # Conseil de pari
    print("╠══════════════════════════════════════════════════════════╣")
    print("║")
    
    result_map = {
        'H': f"🏠 Victoire {home}",
        'D': "🤝 Match Nul",
        'A': f"🏃 Victoire {away}"
    }
    
    result_text = result_map.get(pred['predicted_result'], "Inconnu")
    
    # Niveau de confiance
    conf = pred['confidence'] * 100
    if conf >= 70:
        emoji = "🔥🔥🔥"
        niveau = "TRÈS HAUTE"
        conseil = "PARI RECOMMANDÉ"
    elif conf >= 50:
        emoji = "🔥🔥"
        niveau = "HAUTE"
        conseil = "PARI INTÉRESSANT"
    elif conf >= 40:
        emoji = "🔥"
        niveau = "MODÉRÉE"
        conseil = "PARI RISQUÉ"
    else:
        emoji = "⚠️"
        niveau = "FAIBLE"
        conseil = "MATCH INCERTAIN - PRUDENCE"
    
    print(f"║  💡 CONSEIL : {conseil}")
    print(f"║  🎰 PARI   : {result_text}")
    print(f"║  {emoji} Confiance {niveau} ({conf:.0f}%)")
    
    # Conseil Over/Under
    print("║")
    if pred['predicted_goals'] > 3.0:
        print("║  ⚽ Conseil secondaire : OVER 2.5 buts (match offensif attendu)")
    elif pred['predicted_goals'] < 2.0:
        print("║  ⚽ Conseil secondaire : UNDER 2.5 buts (match fermé attendu)")
    else:
        print("║  ⚽ Conseil secondaire : Buts proches de 2.5 (incertain)")
    
    # Conseil Fautes
    if pred['predicted_fouls'] >= 25:
        print(f"║  🟨 Match tendu attendu ({pred['predicted_fouls']} fautes)")
    
    print("║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────────────────────
# SAUVEGARDE EN BASE
# ─────────────────────────────────────────────────────────

def save_prediction_to_db(pred: dict):
    """Sauvegarde la prédiction dans Supabase."""
    try:
        from src.database import insert_prediction
        
        db_pred = {
            'home_team': pred['home_team'],
            'away_team': pred['away_team'],
            'prob_home_win': pred['prob_home'],
            'prob_draw': pred['prob_draw'],
            'prob_away_win': pred['prob_away'],
            'predicted_result': pred['predicted_result'],
            'predicted_over_under': pred['predicted_goals'],
            'predicted_fouls': pred['predicted_fouls'],
            'confidence_score': pred['confidence'],
            'model_version': 'v1.0',
            'prediction_date': pred['timestamp']
        }
        
        insert_prediction(db_pred)
        print("💾 Prédiction sauvegardée dans Supabase ✅")
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde DB : {e}")


# ─────────────────────────────────────────────────────────
# BOUCLE INTERACTIVE PRINCIPALE
# ─────────────────────────────────────────────────────────

def main():
    """Point d'entrée interactif du script de pronostic."""
    
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        ⚽ PRONOSTIC FOOTBALL IA — Mode Interactif ⚽    ║")
    print("║           Powered by XGBoost + Supabase                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # 1. Charger les cerveaux
    try:
        models = load_brains()
    except FileNotFoundError as e:
        print(f"\n{e}")
        return
    
    # 2. Initialiser le FeatureEngineer (connexion Supabase)
    try:
        fe = FeatureEngineer()
        print("🔗 Connexion Supabase établie\n")
    except Exception as e:
        print(f"❌ Erreur connexion Supabase : {e}")
        return
    
    # 3. Boucle interactive
    while True:
        print("─" * 58)
        print("  Entrez les équipes (ou 'q' pour quitter)")
        print("─" * 58)
        
        # Input : Équipe Domicile
        home_team = input("\n  🏠 Équipe DOMICILE : ").strip()
        if home_team.lower() in ('q', 'quit', 'exit', ''):
            print("\n👋 À bientôt ! Bonne chance avec vos paris !\n")
            break
        
        # Input : Équipe Extérieur
        away_team = input("  🏃 Équipe EXTÉRIEUR : ").strip()
        if away_team.lower() in ('q', 'quit', 'exit', ''):
            print("\n👋 À bientôt ! Bonne chance avec vos paris !\n")
            break
        
        # Input optionnel : Arbitre
        referee = input("  🧑‍⚖️  Arbitre (Entrée pour ignorer) : ").strip()
        referee = referee if referee else None
        
        # Afficher la forme récente
        print(f"\n📡 Récupération de la forme de {home_team}...")
        home_stats = get_team_recent_form(home_team, fe)
        print(f"\n  🏠 {home_team} :")
        display_team_form(home_team, home_stats)
        
        print(f"\n📡 Récupération de la forme de {away_team}...")
        away_stats = get_team_recent_form(away_team, fe)
        print(f"\n  🏃 {away_team} :")
        display_team_form(away_team, away_stats)
        
        # Prédiction
        print("\n🤖 Analyse en cours...")
        try:
            prediction = predict_match(home_team, away_team, models, fe, referee)
            display_prediction(prediction)
            
            # Sauvegarder ?
            save = input("  💾 Sauvegarder dans Supabase ? (o/n) : ").strip().lower()
            if save in ('o', 'oui', 'y', 'yes'):
                save_prediction_to_db(prediction)
            
        except Exception as e:
            print(f"\n❌ Erreur de prédiction : {e}")
            print("   Vérifiez que les noms d'équipes sont corrects.\n")
        
        print()


if __name__ == "__main__":
    main()
