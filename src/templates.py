
# -----------------------------------------------------------------------------
# HTML TEMPLATES (Red & White Sport Theme)
# -----------------------------------------------------------------------------

HERO_TEMPLATE = """<div style="text-align: center; margin-bottom: 40px; padding: 40px 0; background: #ffffff; border-bottom: 1px solid #e2e8f0;">
    <h1 style="font-size: 3.5rem; margin-bottom: 8px; font-weight: 900; letter-spacing: -0.05em; color: #0f172a;">
        BET<span style="color: #dc2626;">APP</span>
    </h1>
    <div style="display: inline-flex; align-items: center; gap: 8px; background: #f1f5f9; padding: 6px 16px; border-radius: 999px;">
        <span style="width: 8px; height: 8px; background: #dc2626; border-radius: 50%; display: block;"></span>
        <span style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            {count} Matches Today
        </span>
    </div>
</div>"""

FEATURED_MATCH_TEMPLATE = """<div style="margin-bottom: 40px;">
<div style="text-transform: uppercase; letter-spacing: 0.1em; color: #dc2626; font-weight: 800; font-size: 0.85rem; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
<span style="font-size: 1.2rem;">🔥</span> MATCH OF THE DAY
</div>
<div class="glass-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; border: none !important; color: white !important; padding: 40px !important; position: relative; overflow: hidden;">
<!-- Decoration -->
<div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(220, 38, 38, 0.2); border-radius: 50%; blur: 40px;"></div>
<div style="text-align: center; margin-bottom: 20px;">
<span style="background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 4px; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: #cbd5e1;">
{league} • {time}
</span>
</div>
<div style="display: flex; align-items: center; justify-content: center; gap: 40px; margin-bottom: 30px;">
<div style="text-align: right; flex: 1;">
<div style="font-size: 2rem; font-weight: 900; line-height: 1; color: white;">{home}</div>
</div>
<img src="{logo_home}" style="width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 0 20px rgba(255,255,255,0.2));">
<div style="font-size: 1.5rem; font-weight: 300; color: #94a3b8; padding: 0 10px;">VS</div>
<img src="{logo_away}" style="width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 0 20px rgba(255,255,255,0.2));">
<div style="text-align: left; flex: 1;">
<div style="font-size: 2rem; font-weight: 900; line-height: 1; color: white;">{away}</div>
</div>
</div>
<div style="text-align: center;">
<div style="display: inline-block; padding: 10px 30px; border: 1px solid rgba(255,255,255,0.2); border-radius: 99px; font-size: 0.9rem; color: #e2e8f0; font-weight: 600;">
AI PREDICTION AVAILABLE
</div>
</div>
</div>
</div>"""

LEAGUE_HEADER_TEMPLATE = """<div style="margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
<div style="font-size: 1.25rem; font-weight: 900; color: #0f172a; text-transform: uppercase; letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px;">
<span style="width: 6px; height: 24px; background: #dc2626; display: block; border-radius: 2px;"></span>
{league}
</div>
<div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">{count} Matches</div>
</div>"""

MATCH_ROW_TEMPLATE = """<div class="match-card" style="background: white; border-radius: 8px; padding: 16px 24px; margin-bottom: 12px; border: 1px solid #f1f5f9; display: flex; align-items: center; justify-content: space-between; transition: all 0.2s; cursor: pointer;">
<div style="width: 80px; font-size: 0.9rem; font-weight: 700; color: #64748b; font-family: 'JetBrains Mono';">
{time}
</div>
<div style="flex: 1; display: flex; align-items: center; gap: 30px;">
<div style="flex: 1; display: flex; align-items: center; justify-content: flex-end; gap: 12px;">
<span style="font-weight: 700; font-size: 1.1rem; color: #0f172a; text-align: right;">{home}</span>
<img src="{logo_home}" style="width: 32px; height: 32px; object-fit: contain;">
</div>
<div style="font-size: 0.8rem; color: #cbd5e1; font-weight: 700;">•</div>
<div style="flex: 1; display: flex; align-items: center; justify-content: flex-start; gap: 12px;">
<img src="{logo_away}" style="width: 32px; height: 32px; object-fit: contain;">
<span style="font-weight: 700; font-size: 1.1rem; color: #0f172a; text-align: left;">{away}</span>
</div>
</div>
<div style="width: 80px; text-align: right;">
<span style="font-size: 0.8rem; color: #dc2626; font-weight: 700; background: #fef2f2; padding: 4px 12px; border-radius: 4px;">PICS</span>
</div>
</div>"""

DASHBOARD_HEADER_TEMPLATE = """<div style="text-align: center; margin-bottom: 40px; padding: 20px; background: white; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
<h2 style="margin: 0; font-size: 1.8rem; color: #0f172a; font-weight: 800; text-transform: uppercase; letter-spacing: -0.02em;">Match Intelligence</h2>
<div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 10px; color: #64748b; font-weight: 500;">
<span>{home}</span>
<span style="color: #dc2626; font-weight: 900;">vs</span>
<span>{away}</span>
</div>
</div>"""

OUTCOME_CARD_TEMPLATE = """<div class="insight-card" style="text-align: center; padding: 40px; border-top: 6px solid #dc2626;">
<div style="font-size: 0.85rem; color: #64748b; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 16px; text-transform: uppercase;">Predicted Outcome</div>
<div style="font-size: 3.5rem; font-weight: 900; margin-bottom: 16px; letter-spacing: -0.03em; line-height: 1; color: #0f172a;" class="{color_class}">
{outcome}
</div>
<div style="display: inline-flex; align-items: center; gap: 8px; background: #f1f5f9; padding: 8px 16px; border-radius: 999px;">
<span style="color: #64748b; font-size: 0.9rem; font-weight: 600;">CONFIDENCE</span>
<span style="color: #0f172a; font-weight: 800; font-size: 1rem;">{confidence:.1f}%</span>
</div>
</div>"""

METRIC_BOX_TEMPLATE = """<div style="background: #f8fafc; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #e2e8f0; height: 100%; display: flex; flex-direction: column; justify-content: center;">
<div style="font-size: 0.75rem; color: #64748b; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">{label}</div>
<div style="font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1.1; letter-spacing: -0.02em;">{value}</div>
<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px; font-weight: 500;">{subtext}</div>
</div>"""

FACTOR_TEMPLATE = """<div style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #dc2626; margin-bottom: 10px; font-size: 0.95rem; display: flex; align-items: center; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); border: 1px solid #f1f5f9;">
<span style="color: #334155; font-weight: 500;">{factor}</span>
</div>"""

AI_INSIGHT_TEMPLATE = """<div style="margin-bottom: 25px; padding: 24px; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-left: 6px solid #8b5cf6;">
<h3 style="margin: 0 0 12px 0; font-size: 1.1rem; color: #1e293b; display: flex; align-items: center; gap: 10px; font-weight: 800;">
<span style="font-size: 1.4rem;">🤖</span>
AI Analysis
</h3>
<p style="color: #475569; line-height: 1.6; font-size: 1rem; margin: 0;">{analysis}</p>
</div>"""

COMPARISON_BAR_TEMPLATE = """<div style="margin-bottom: 24px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
<span style="color: #0f172a;">{left_val}</span>
<span>{title}</span>
<span style="color: #0f172a;">{right_val}</span>
</div>
<div style="display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: #e2e8f0;">
<div style="width: {left_pct}%; background: #0f172a;"></div>
<div style="flex: 1; background: transparent;"></div>
<div style="width: {right_pct}%; background: #dc2626;"></div>
</div>
</div>"""
