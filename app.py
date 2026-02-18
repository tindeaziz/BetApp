"""
💎 BetApp — Premium AI Football Analytics
==========================================

High-end "SaaS-style" interface for AI-powered football predictions.
Features "Midnight Blue" dark theme, glassmorphism, and neon accents.
"""

import streamlit as st
import os
import sys
import time
import textwrap
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import get_todays_matches, ingest_upcoming_fixtures, get_cached_prediction
from src.prediction import load_models, run_prediction
from src.templates import (
    HERO_TEMPLATE, 
    MATCH_ROW_TEMPLATE, 
    DASHBOARD_HEADER_TEMPLATE, 
    OUTCOME_CARD_TEMPLATE, 
    METRIC_BOX_TEMPLATE, 
    FACTOR_TEMPLATE, 
    AI_INSIGHT_TEMPLATE, 
    COMPARISON_BAR_TEMPLATE,
    FEATURED_MATCH_TEMPLATE,
    LEAGUE_HEADER_TEMPLATE
)

# Load environment
load_dotenv()

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="BetApp AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SESSION STATE ──
if "selected_match_id" not in st.session_state:
    st.session_state.selected_match_id = None

# ─────────────────────────────────────────────────────
# ADVANCED CSS (DARK MODE / GLASSMORPHISM)
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* 1. FONTS & RESET */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* 2. BACKGROUND & MAIN CONTAINER */
    .stApp {
        background: #f8fafc !important; /* Slate 50 */
        color: #0f172a !important; /* Slate 900 */
    }
    
    /* 3. CARDS & CONTAINERS */
    .glass-card, .metric-box {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important; /* Slate 200 */
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        color: #0f172a !important;
    }
    
    .glass-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025) !important;
    }
    
    /* 4. TYPOGRAPHY & ACCENTS */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    .stMarkdown p {
        color: #334155 !important;
    }
    
    .text-red {
        color: #dc2626 !important; /* Red 600 */
    }
    
    .text-slate {
        color: #64748b !important; /* Slate 500 */
    }

    /* 5. BUTTONS (The Red Power Button) */
    .stButton > button {
        background: #dc2626 !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.3) !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #b91c1c !important; /* Red 700 */
        box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.4) !important;
        color: white !important;
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Secondary Button (Back) */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #f1f5f9 !important;
        color: #475569 !important;
        box-shadow: none !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* 6. PROGRESS BARS */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ef4444, #dc2626) !important; /* Red Gradient */
    }
    
    /* 7. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    div[data-testid="stSidebar"] h3 {
         color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# MAIN VISUALIZATION
# ─────────────────────────────────────────────────────
from src.templates import (
    HERO_TEMPLATE, 
    MATCH_ROW_TEMPLATE, 
    DASHBOARD_HEADER_TEMPLATE, 
    OUTCOME_CARD_TEMPLATE, 
    METRIC_BOX_TEMPLATE, 
    FACTOR_TEMPLATE,
    AI_INSIGHT_TEMPLATE,
    COMPARISON_BAR_TEMPLATE
)

def render_hero(todays_count):
    st.markdown(HERO_TEMPLATE.format(count=todays_count), unsafe_allow_html=True)


def render_match_row(match):
    """Render a single match row with glassmorphism style."""
    m_id = match['id']
    home = match['home']
    away = match['away']
    time_label = match['label'].split('—')[0].strip()
    
    logo_h = match.get('home_logo') or "https://media.api-sports.io/football/teams/default.png"
    logo_a = match.get('away_logo') or "https://media.api-sports.io/football/teams/default.png"
    
    # HTML Layout for the row
    st.markdown(MATCH_ROW_TEMPLATE.format(
        time=time_label,
        league=match['league'],
        home=home,
        away=away,
        logo_home=logo_h,
        logo_away=logo_a
    ), unsafe_allow_html=True)
    
    # Action Button (Streamlit limitation: must be outside HTML block)
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button(f"ANALYZE MATCH ⚡", key=f"btn_{m_id}"):
            st.session_state.selected_match_id = m_id
            st.rerun()


def render_analytics_dashboard(match, models):
    """Render the detailed analytics view."""
    
    # Header
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅ BACK", use_container_width=True):
            st.session_state.selected_match_id = None
            st.rerun()
            
    st.markdown(DASHBOARD_HEADER_TEMPLATE.format(
        home=match['home'], 
        away=match['away']
    ), unsafe_allow_html=True)
    
    # Run Prediction
    with st.spinner("Running Ensemble Models..."):
        try:
            from src.database import get_cached_prediction, save_prediction_cache
            
            # Check cache first
            cached = get_cached_prediction(match['id'])
            if cached:
                pred = cached
            else:
                pred = run_prediction(
                    match['home'], match['away'],
                    models=models,
                    referee=match.get('referee'),
                    league=match.get('league'),
                    league_id=match.get('league_id'),
                    match_date=match.get('match_date')
                )
                # Save to cache for next time
                try:
                    save_prediction_cache(match['id'], pred)
                except Exception as e:
                    print(f"Cache miss save failed: {e}")

            
            # 1. MAIN PREDICTION CARD
            cls = pred['predicted_result']
            confidence = pred['confidence'] * 100
            
            outcome_map = {'H': f"{match['home']} WIN", 'D': "DRAW", 'A': f"{match['away']} WIN"}
            outcome_text = outcome_map.get(cls, "UNKNOWN")
            
            # Color logic
            color_class = "neon-text-blue"
            if confidence < 50: color_class = "text-warning" # fallback
            
            st.markdown(OUTCOME_CARD_TEMPLATE.format(
                outcome=outcome_text,
                confidence=confidence,
                color_class=color_class
            ), unsafe_allow_html=True)
            
            # 2. PROBABILITIES & EXPECTED METRICS
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{match["home"]} WIN</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pred["prob_home"]:.1%}</div>', unsafe_allow_html=True)
                st.progress(pred['prob_home'])
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">DRAW</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pred["prob_draw"]:.1%}</div>', unsafe_allow_html=True)
                st.progress(pred['prob_draw'])
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{match["away"]} WIN</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pred["prob_away"]:.1%}</div>', unsafe_allow_html=True)
                st.progress(pred['prob_away'])
                st.markdown('</div>', unsafe_allow_html=True)
            

            # DOUBLE CHANCE
            st.markdown("### 🛡️ Double Chance")
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label=f"{match['home']} / DRAW",
                    value=f"{pred.get('prob_1x', 0):.1%}",
                    color="#cbd5e1",
                    subtext="1X"
                ), unsafe_allow_html=True)
            with dc2:
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label=f"{match['home']} / {match['away']}",
                    value=f"{pred.get('prob_12', 0):.1%}",
                    color="#cbd5e1",
                    subtext="12"
                ), unsafe_allow_html=True)
            with dc3:
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label=f"DRAW / {match['away']}",
                    value=f"{pred.get('prob_x2', 0):.1%}",
                    color="#cbd5e1",
                    subtext="X2"
                ), unsafe_allow_html=True)

            # ──────────────────────────────────────────────
            # NEW: CORRECT SCORE & ADVANCED MARKETS
            # ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            c_score, c_markets = st.columns([1, 1])
            
            with c_score:
                st.markdown("### 🎯 Top Correct Scores")
                scores = pred.get('correct_score_probs', {})
                # Get Top 3
                top_scores = list(scores.items())[:3]
                
                for sc, p in top_scores:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; background: #f1f5f9; padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; border: 1px solid #e2e8f0;">
                        <span style="font-weight: bold; color: #0f172a;">{sc}</span>
                        <span style="color: #0284c7; font-family: monospace; font-weight: 600;">{p:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with c_markets:
                st.markdown("### 📊 Market Probabilities")
                
                # BTTS
                btts = pred.get('btts_prob', 0.5)
                st.markdown(f"**BTTS (Both Teams to Score)**")
                st.progress(btts)
                st.caption(f"Yes: {btts:.1%} | No: {1-btts:.1%}")
                
                # Over 2.5
                ou = pred.get('over_under_probs', {})
                o25 = ou.get('over_2_5', 0.5)
                st.markdown(f"**Over 2.5 Goals**")
                st.progress(o25)
                st.caption(f"Over: {o25:.1%} | Under: {1-o25:.1%}")

            # ──────────────────────────────────────────────
            # NEW: FOUL & CARD ANALYTICS (DIRTY GAME MONITOR)
            # ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🟨 Dirty Game Monitor (Fouls & Cards)")
            
            foul_data = pred.get('foul_analytics', {})
            intensity = foul_data.get('intensity_score', 1.0)
            
            # Intensity Badge
            if intensity > 1.3:
                badge = "🔥 HIGH TENSION"
                badge_color = "#ef4444" # red
            elif intensity > 1.1:
                badge = "⚡ MODERATE"
                badge_color = "#f59e0b" # amber
            else:
                badge = "😌 CALM"
                badge_color = "#10b981" # green
                
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label="PREDICTED FOULS",
                    value=f"{pred['predicted_fouls']}",
                    color="#c084fc",
                    subtext=f"Intensity: {intensity:.2f}"
                ), unsafe_allow_html=True)
                
            with col_f2:
                cards = foul_data.get('expected_cards', 0)
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label="EXPECTED CARDS",
                    value=f"{cards}",
                    color="#f472b6",
                    subtext=f"Ref Strictness: {foul_data.get('referee_strictness', 1.0):.2f}x"
                ), unsafe_allow_html=True)
                
            with col_f3:
                card_prob = foul_data.get('prob_over_3_5_cards', 0)
                st.markdown(METRIC_BOX_TEMPLATE.format(
                    label="OVER 3.5 CARDS",
                    value=f"{card_prob:.1%}",
                    color="#fbbf24",
                    subtext="Probability"
                ), unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px; padding: 6px; background: {badge_color}33; border: 1px solid {badge_color}; border-radius: 8px; color: {badge_color}; font-weight: bold;">
                {badge} MATCH
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            feats = pred.get('features', {})
            
            # AI INSIGHT
            if 'explanation' in pred:
                st.markdown(AI_INSIGHT_TEMPLATE.format(analysis=pred['explanation']), unsafe_allow_html=True)

            # COMPARISON BARS
            st.markdown('<div style="margin-top: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
            
            # 1. Form
            h_form = feats.get('home_form', 0.5) * 100
            a_form = feats.get('away_form', 0.5) * 100
            f_tot = h_form + a_form if (h_form + a_form) > 0 else 1
            st.markdown(COMPARISON_BAR_TEMPLATE.format(
                left_val=f"{h_form:.0f}%", 
                title="RECENT FORM", 
                right_val=f"{a_form:.0f}%", 
                left_pct=(h_form/f_tot)*100, right_pct=(a_form/f_tot)*100
            ), unsafe_allow_html=True)
            
            # 2. Attack
            h_att = feats.get('home_offensive', 1.0)
            a_att = feats.get('away_offensive', 1.0)
            a_tot = h_att + a_att if (h_att + a_att) > 0 else 1
            st.markdown(COMPARISON_BAR_TEMPLATE.format(
                left_val=f"{h_att:.2f}", 
                title="ATTACK (Goals/Match)", 
                right_val=f"{a_att:.2f}", 
                left_pct=(h_att/a_tot)*100, right_pct=(a_att/a_tot)*100
            ), unsafe_allow_html=True)
            
            # 3. Defense (Inverted visual: Lower is better, but bar size usually represents magnitude)
            # Let's show magnitude of "Weakness" (Concended)
            h_def = feats.get('home_defensive', 1.0)
            a_def = feats.get('away_defensive', 1.0)
            d_tot = h_def + a_def if (h_def + a_def) > 0 else 1
            st.markdown(COMPARISON_BAR_TEMPLATE.format(
                left_val=f"{h_def:.2f}", 
                title="DEFENSE (Conceded/Match)", 
                right_val=f"{a_def:.2f}", 
                left_pct=(h_def/d_tot)*100, right_pct=(a_def/d_tot)*100
            ), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. GOALS & FOULS (Secondary Metrics) - REMOVED (Redundant with new sections)

                
            # 4. KEY FACTORS
            st.markdown("### 🧬 Key Factors")
            
            feats = pred['features']
            factors = []
            
            if feats.get('home_form', 0) > 0.6: factors.append(f"📈 {match['home']} is in great form")
            if feats.get('h2h_home_wins', 0) > 0.5: factors.append(f"🦁 {match['home']} dominates H2H")
            if feats.get('offensive_diff', 0) > 0.5: factors.append(f"⚔️ High offensive mismatch")
            
            if not factors: factors.append("⚖️ Balanced matchup, no strong dominating factors.")
            
            for f in factors:
                st.markdown(FACTOR_TEMPLATE.format(factor=f), unsafe_allow_html=True)
            
            # Raw Data Expander
            with st.expander("🛠️ View Raw Model Inputs"):
                st.json(feats)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            # st.exception(e) # Debug


# ─────────────────────────────────────────────────────
# APP ENTRY POINT
# ─────────────────────────────────────────────────────

def main():
    # Load Models
    models = load_models()
    if models is None:
        st.error("SYSTEM ERROR: AI Models Offline. Run training pipeline.")
        return

    # Data Ingestion
    todays = get_todays_matches()
    
    # Render
    if st.session_state.selected_match_id:
        match = next((m for m in todays if m['id'] == st.session_state.selected_match_id), None)
        if match:
            render_analytics_dashboard(match, models)
        else:
            st.session_state.selected_match_id = None
            st.rerun()
    else:
        render_hero(len(todays))
        
        # Filters (Sidebar)
        with st.sidebar:
            st.markdown("### ⚙️ Control Panel")
            if st.button("↻ Refresh Data", use_container_width=True):
                with st.spinner("Syncing..."):
                    try:
                        # Default ingest (Premier League for demo)
                        result = ingest_upcoming_fixtures(39, 2025, next_n=20)
                        st.success(f"Synced {result} matches.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sync failed: {e}")
        
        # Match List
    # Featured & Grouped List
    if not todays:
         st.info("No matches confirmed for today. Check Control Panel to sync fixtures.")
    else:
        # Helper for safe data
        def get_safe_match_data(m):
            # Time
            t = m.get('time', '00:00')
            if t == '00:00': t = 'TBD'
            
            # Logos
            lh = m.get('logo_home')
            if not lh or str(lh) in ['0', '0.png'] or '0.png' in str(lh):
                lh = "https://placehold.co/60x60/png?text=?"
            
            la = m.get('logo_away')
            if not la or str(la) in ['0', '0.png'] or '0.png' in str(la):
                la = "https://placehold.co/60x60/png?text=?"
                
            return t, lh, la

        # 1. FEATURED MATCH
        featured = todays[0]
        f_time, f_logo_h, f_logo_a = get_safe_match_data(featured)
        
        # Featured Match Container
        with st.container(border=True):
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 10px;">
                <span style="background: #fee2e2; color: #dc2626; padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.1em;">
                    🔥 MATCH OF THE DAY
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            c_f1, c_f2, c_f3 = st.columns([2, 3, 2], vertical_alignment="center")
            
            with c_f1:
                st.markdown(f"<div style='text-align: right; font-size: 1.5rem; font-weight: 900;'>{featured['home']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: right; display: flex; justify-content: flex-end;'><img src='{f_logo_h}' style='height: 80px;'></div>", unsafe_allow_html=True)
                
            with c_f2:
                st.markdown(f"<div style='text-align: center; font-size: 2rem; font-weight: 300; color: #94a3b8;'>VS</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-weight: 700; color: #64748b;'>{f_time}</div>", unsafe_allow_html=True)
                
            with c_f3:
                 st.markdown(f"<div style='text-align: left; font-size: 1.5rem; font-weight: 900;'>{featured['away']}</div>", unsafe_allow_html=True)
                 st.markdown(f"<div style='text-align: left; display: flex; justify-content: flex-start;'><img src='{f_logo_a}' style='height: 80px;'></div>", unsafe_allow_html=True)

            st.write("")
            if st.button(f"⚡ ANALYZE MATCH", key=f"btn_feat_{featured['id']}", use_container_width=True, type="primary"):
                st.session_state.selected_match_id = featured['id']
                st.rerun()

        # 2. MATCH LIST (Grouped)
        st.markdown("<br>", unsafe_allow_html=True)
        from itertools import groupby
        matches_sorted = sorted(todays, key=lambda x: x.get('league', 'Unknown'))
        
        for league, league_matches in groupby(matches_sorted, key=lambda x: x.get('league', 'Unknown')):
            league_matches_list = list(league_matches)
             # League Header
            st.markdown(LEAGUE_HEADER_TEMPLATE.format(league=league, count=len(league_matches_list)), unsafe_allow_html=True)
            
            for match in league_matches_list:
                if match['id'] == featured['id']: continue
                
                # Get safe data
                m_time, m_logo_h, m_logo_a = get_safe_match_data(match)

                with st.container(border=True):
                    c_time, c_match, c_btn = st.columns([1.2, 5, 2], vertical_alignment="center")
                    
                    with c_time:
                        st.markdown(f"**{m_time}**", unsafe_allow_html=True)
                        if match.get('status') in ['LIVE', '1H', '2H', 'HT']:
                             st.markdown("🔴 *Live*", unsafe_allow_html=True)

                    with c_match:
                        c_h_name, c_h_logo, c_sep, c_a_logo, c_a_name = st.columns([3, 0.8, 0.4, 0.8, 3], vertical_alignment="center")
                        with c_h_name: st.markdown(f"<div style='text-align: right; font-weight: 700; color: #0f172a;'>{match['home']}</div>", unsafe_allow_html=True)
                        with c_h_logo: st.image(m_logo_h, width=32)
                        with c_sep: st.markdown("<div style='text-align: center; color: #cbd5e1;'>•</div>", unsafe_allow_html=True)
                        with c_a_logo: st.image(m_logo_a, width=32)
                        with c_a_name: st.markdown(f"<div style='text-align: left; font-weight: 700; color: #0f172a;'>{match['away']}</div>", unsafe_allow_html=True)

                    with c_btn:
                        if st.button("⚡ ANALYZE", key=f"btn_{match['id']}", use_container_width=True):
                            st.session_state.selected_match_id = match['id']
                            st.rerun()


if __name__ == "__main__":
    main()
