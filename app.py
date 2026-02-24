""""""" Aspect-Based Sentiment Analysis of Student Feedback """""""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction import (
    get_model_and_vectorizer, analyze_feedback,
    analyze_batch, compute_summary_stats, SENTIMENT_COLORS
)
from aspect_extraction import get_all_aspects

def conf_to_scale(confidence: float, sentiment: str) -> int:
    if sentiment == "Neutral":
        return 5
    elif sentiment == "Positive":
        scaled = 6 + round((confidence - 0.33) / (1.0 - 0.33) * 4)
        return max(6, min(10, scaled))
    else:
        scaled = 4 - round((confidence - 0.33) / (1.0 - 0.33) * 3)
        return max(1, min(4, scaled))

st.set_page_config(
    page_title="SentimentIQ — Student Feedback Analysis",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""<style>
:root {
  --bg:     #05050d;
  --g1:     rgba(255,255,255,0.038);
  --g2:     rgba(255,255,255,0.065);
  --b0:     rgba(255,255,255,0.07);
  --bv:     rgba(124,58,237,0.55);
  --purple: #7c3aed;
  --violet: #a78bfa;
  --glow2:  rgba(124,58,237,0.38);
  --tx:     #edeaf8;
  --tx2:    #9490b0;
  --tx3:    #4c4966;
  --green:  #34d399;
  --red:    #f87171;
  --amber:  #fbbf24;
  --r:      14px;
  --r2:     20px;
  --font:   'Helvetica Neue', Helvetica, Arial, sans-serif;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html,body,[class*="css"] {
  font-family:var(--font)!important;
  background:var(--bg)!important;
  color:var(--tx)!important;
  -webkit-font-smoothing:antialiased;
}
.main { background:var(--bg)!important; }
.block-container { padding:0 3rem 7rem!important; max-width:1180px!important; }
[data-testid="stSidebar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display:none!important; }
#MainMenu,footer,header { visibility:hidden!important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(124,58,237,0.3); border-radius:4px; }

/* ── TOPBAR ── */
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  height:62px; border-bottom:1px solid var(--b0); margin-bottom:0;
  /* no sticky, no backdrop — just a clean brand bar */
}
.topbar-brand {
  display:flex; align-items:center; gap:10px;
  font-size:0.97rem; font-weight:700; color:#fff; letter-spacing:-0.02em;
}
.topbar-icon {
  width:32px; height:32px; border-radius:9px;
  background:linear-gradient(135deg,#7c3aed,#a78bfa);
  display:flex; align-items:center; justify-content:center;
  font-size:0.85rem; font-weight:900; color:#fff;
}
.topbar-brand em { font-style:normal; color:var(--violet); }
.topbar-status {
  display:flex; align-items:center; gap:7px;
  background:rgba(124,58,237,0.09); border:1px solid rgba(124,58,237,0.22);
  border-radius:50px; padding:6px 16px;
  font-size:0.73rem; font-weight:600; color:var(--violet); letter-spacing:0.03em;
}
.topbar-dot {
  width:6px; height:6px; border-radius:50%;
  background:var(--green); box-shadow:0 0 7px var(--green);
  animation:blink 2.2s ease infinite;
}
@keyframes blink { 0%,100%{opacity:1} 55%{opacity:0.35} }

/* ── HERO ── */
.hero {
  text-align:center; padding:5.5rem 2rem 0;
  position:relative; overflow:hidden;
}
.hero-orb{
  position:absolute;
  top:-120px;
  left:50%;
  transform:translateX(-50%);
  width:900px;
  height:560px;
  pointer-events:none;
  border-radius:50%;
  background:radial-gradient(ellipse at 50% 32%,
    rgba(124,58,237,0.28) 0%,
    rgba(99,40,235,0.11) 40%,
    transparent 68%);
  mask-image:radial-gradient(circle at center, black 65%, transparent 100%);
}
.hero-tag {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(124,58,237,0.10); border:1px solid rgba(124,58,237,0.26);
  border-radius:50px; padding:5px 16px;
  font-size:0.68rem; font-weight:700; color:var(--violet);
  letter-spacing:0.07em; text-transform:uppercase;
  margin-bottom:1.8rem; animation:up 0.7s ease both;
}
.hero-tag-dot { width:5px; height:5px; border-radius:50%; background:var(--green); flex-shrink:0; }
.hero-h1{
  font-size:clamp(2.6rem,4.5vw,3.9rem);
  font-weight:800;
  color:#fff;
  line-height:1.08;
  letter-spacing:-0.05em;
  margin-bottom:1.4rem;
  animation:up 0.7s 0.07s ease both;

  white-space:nowrap;

  white-space:nowrap;

  display:inline-block;     /* break from text-align centering */
  position:relative;
  left:-30px; 
}
.hero-words {
  display:flex; justify-content:center; gap:2.5rem;
  margin-bottom:1rem; animation:up 0.7s 0.14s ease both;
}
.hero-words em {
  font-style:normal; font-size:clamp(2rem,3vw,3.9rem); font-weight:700;
  background:linear-gradient(130deg,#c4b5fd 0%,#a78bfa 45%,#7c3aed 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  letter-spacing:-0.02em;
            margin-bottom:1rem;
}
@keyframes up { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }

/* ── TRUST STRIP ── */
            
.strip {
  display: flex;
  justify-content: center;
  align-items: stretch;
  gap: 1.2rem;                 
  margin: 2rem 0;
  padding: 0;
  animation: up 0.7s 0.21s ease both;
}
            
.strip-item {
  flex: 1;
  min-width: 180px;
  max-width: 220px;
  background:var(--g1); 
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  padding: 1.5rem 1rem;

  border: 1px solid var(--b0);
  border-radius: 16px;

  transition: transform 0.2s ease, box-shadow 0.2s ease, color 0.2s ease;
}
            
.strip-item:last-child { border-right:none; }
.strip-item:hover { background:var(--g2); }
.strip-num { font-size:1.55rem; font-weight:800; color:#fff; letter-spacing:-0.04em; line-height:1; }
.strip-lbl { font-size:0.66rem; font-weight:700; color:var(--tx3); letter-spacing:0.1em; text-transform:uppercase; margin-top:5px; }


.strip-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.08);
  background:rgba(124,58,237,0.10);
}

.strip-item:hover .strip-num,
.strip-item:hover .strip-lbl {
  background: radial-gradient(ellipse at 50% 32%,
    rgba(255,255,255,0.95) 0%,
    rgba(226,214,255,0.9) 35%,
    rgba(196,181,253,0.65) 60%,
    rgba(167,139,250,0.35) 80%,
    transparent 100%
  );

  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* ── FEATURE CARDS ── */
/* st.columns sets each col to position:relative; we override to make cards equal height */
[data-testid="column"] { display:flex!important; flex-direction:column!important; }
.fcard-outer {
  background:var(--g1); border:1px solid var(--b0);
  border-radius:var(--r2); overflow:hidden;
  position:relative;
  transition:border-color 0.3s, transform 0.3s, background 0.3s;
  display:flex; flex-direction:column;
  flex:1;  /* fill the column height so both cards are equal */
}
.fcard-outer::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(167,139,250,0.5),transparent);
  z-index:1;
}
.fcard-outer::after {
  content:''; position:absolute; bottom:-80px; right:-80px;
  width:220px; height:220px; border-radius:50%; pointer-events:none;
  background:radial-gradient(circle,rgba(124,58,237,0.09),transparent 70%);
}
.fcard-outer:hover {
  border-color:var(--bv); background:var(--g2);
  transform:translateY(-4px);
  box-shadow:0 28px 60px rgba(124,58,237,0.10);
}
.fcard-body {
  padding:2.4rem 2.6rem 0;
  flex:1;
}
.fcard-ico {
  width:44px; height:44px; border-radius:12px;
  background:linear-gradient(135deg,rgba(124,58,237,0.25),rgba(124,58,237,0.07));
  border:1px solid rgba(124,58,237,0.22);
  display:flex; align-items:center; justify-content:center;
  font-size:1.1rem; margin-bottom:1.3rem;
}
.fcard-eyebrow { font-size:0.62rem; font-weight:700; color:var(--violet); letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.45rem; }
.fcard-title { font-size:1.2rem; font-weight:700; color:#fff; letter-spacing:-0.025em; margin-bottom:0.55rem; line-height:1.3; }
.fcard-desc { font-size:0.86rem; color:var(--tx2); line-height:1.78; margin-bottom:1.3rem; }
.fcard-chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1.8rem; }
.chip { background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.20); color:var(--violet); padding:3px 11px; border-radius:50px; font-size:0.71rem; font-weight:600; letter-spacing:0.02em; }
.fcard-foot { padding:0 2.2rem 2.2rem; }
/* button inside card — remove extra streamlit spacing */
.fcard-foot [data-testid="stButton"],
.fcard-foot .stButton { margin-top:0!important; }

/* ── PAGE HEADER (inner pages) ── */
.pghdr {
  display:flex; align-items:flex-end; justify-content:space-between;
  padding:2.6rem 0 1.8rem; border-bottom:1px solid var(--b0); margin-bottom:2.4rem;
}
.pghdr-title { font-size:2rem; font-weight:800; color:#fff; letter-spacing:-0.045em; line-height:1; }
.pghdr-sub { font-size:0.87rem; color:var(--tx2); margin-top:6px; }
.pghdr-badge { background:rgba(124,58,237,0.10); border:1px solid rgba(124,58,237,0.22); color:var(--violet); padding:7px 16px; border-radius:50px; font-size:0.73rem; font-weight:700; flex-shrink:0; }

/* ── SECTION LABEL ── */
.slbl,.section-header {
  font-size:0.61rem; font-weight:700; color:var(--violet);
  letter-spacing:0.14em; text-transform:uppercase;
  padding-bottom:0.65rem; border-bottom:1px solid var(--b0); margin:2.2rem 0 1.3rem;
}

/* ── INPUT ZONE ── */
[data-baseweb="textarea"] {
  background: var(--g1) !important;
  border-radius: var(--r2);
}

[data-baseweb="textarea"] textarea {
  background: transparent !important;
}

.izone::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(167,139,250,0.55),transparent);
}
.izone:focus-within {
  border-color:rgba(124,58,237,0.46);
  box-shadow:0 0 0 4px rgba(124,58,237,0.07),0 24px 60px rgba(124,58,237,0.08);
}
.izone-eyebrow { font-size:0.6rem; font-weight:700; color:var(--violet); letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.9rem; }
            

div[data-testid="stButton"] button[kind="secondary"]{
  width:100%;
  height:46px;              /* comfortable height */
  display:flex;
  align-items:center;
  justify-content:center;
  padding:0 12px;           /* horizontal breathing space */
  border-radius:999px;
  font-size:0.82rem;
}
            

/* ── RESULT BANNER ── */
.rbanner { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r2); padding:2rem 2.2rem; margin:3rem 0; position:relative; overflow:hidden; animation:up 0.45s ease both; }
.rbanner::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(167,139,250,0.48),transparent); }
.rbanner-lbl { font-size:0.6rem; font-weight:700; color:var(--violet); letter-spacing:0.13em; text-transform:uppercase; padding-bottom:0.7rem; border-bottom:1px solid var(--b0); margin-bottom:1.3rem; }
.rgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }
.rtile { background:rgba(255,255,255,0.022); border:1px solid var(--b0); border-radius:12px; padding:1.2rem 0.9rem; text-align:center; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:7px; }
.rtile-lbl { font-size:0.65rem; font-weight:700; color:var(--tx3); letter-spacing:0.1em; text-transform:uppercase; }
.rtile-val { font-size:2.1rem; font-weight:800; color:#fff; letter-spacing:-0.05em; line-height:1; }
.rtile-sub { font-size:0.75rem; color:var(--tx3); }

/* ── INSIGHT PANELS ── */
.ipanel { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r); padding:1.6rem 1.8rem; }
.ipanel-lbl { font-size:0.61rem; font-weight:700; color:var(--violet); letter-spacing:0.13em; text-transform:uppercase; padding-bottom:0.75rem; border-bottom:1px solid var(--b0); margin-bottom:1.1rem; }

/* ── ASPECT CARDS ── */
.acard { display:flex; align-items:center; justify-content:space-between; background:rgba(255,255,255,0.022); border:1px solid var(--b0); border-left:2px solid var(--purple); border-radius:10px; padding:0.85rem 1.1rem; margin-bottom:0.55rem; transition:all 0.2s; }
.acard:hover { border-left-color:var(--violet); background:var(--g2); transform:translateX(3px); }
.acard-name { font-size:0.87rem; font-weight:600; color:var(--tx); }
.acard-score { font-size:0.72rem; color:var(--tx3); margin-top:2px; }
.abar-wrap { width:100%; background:rgba(255,255,255,0.05); border-radius:2px; height:2px; margin-top:7px; overflow:hidden; }
.abar { height:100%; border-radius:2px; animation:grow 0.85s cubic-bezier(0.4,0,0.2,1) both; }
@keyframes grow { from{width:0!important} }

/* ── BADGES ── */
.sentiment-badge { display:inline-flex; align-items:center; gap:5px; padding:5px 13px; border-radius:50px; font-weight:600; font-size:0.79rem; }
.badge-positive { background:rgba(52,211,153,0.10);  color:var(--green); border:1px solid rgba(52,211,153,0.25); }
.badge-negative { background:rgba(248,113,113,0.10); color:var(--red);   border:1px solid rgba(248,113,113,0.25); }
.badge-neutral  { background:rgba(251,191,36,0.10);  color:var(--amber); border:1px solid rgba(251,191,36,0.25); }

/* ── METRIC CARDS ── */
.mc,.metric-card { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r); padding:1.5rem 1rem; text-align:center; position:relative; overflow:hidden; transition:all 0.28s; }
.mc:hover,.metric-card:hover { background:var(--g2); border-color:var(--bv); transform:translateY(-4px); box-shadow:0 20px 50px rgba(124,58,237,0.12); }
.mc::after,.metric-card::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.mc.positive::after { background:var(--green); }
.mc.negative::after { background:var(--red); }
.mc.neutral::after  { background:var(--amber); }
.mc.total::after    { background:linear-gradient(90deg,var(--purple),var(--violet)); }
.mc-val,.metric-value { font-size:2.5rem; font-weight:800; color:#fff; line-height:1; letter-spacing:-0.05em; }
.mc-lbl,.metric-label { font-size:0.64rem; font-weight:700; color:var(--tx3); letter-spacing:0.1em; text-transform:uppercase; margin-top:4px; }

/* ── CHART PANEL ── */
.cpanel { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r); padding:1.3rem 1rem 0.3rem; }

/* ── BATCH INTRO GRID ── */
.bgrid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:2.4rem; }
.bpanel { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r); padding:1.8rem 2rem; position:relative; overflow:hidden; }
.bpanel::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(167,139,250,0.38),transparent); }
.bpanel-lbl { font-size:0.61rem; font-weight:700; color:var(--violet); letter-spacing:0.13em; text-transform:uppercase; margin-bottom:0.9rem; }

/* ── ABOUT GRID ── */
.agrid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1.6rem 0; }
.apanel { background:var(--g1); border:1px solid var(--b0); border-radius:var(--r); padding:1.8rem 2rem; }
.apanel h3 { font-size:0.61rem; font-weight:700; color:var(--violet); letter-spacing:0.13em; text-transform:uppercase; padding-bottom:0.7rem; border-bottom:1px solid var(--b0); margin-bottom:1.1rem; }
.apanel p,.apanel li { font-size:0.87rem; color:var(--tx2); line-height:1.82; }
.apanel li { margin-left:1.1rem; margin-bottom:0.3rem; }
.apanel b { color:var(--tx); font-weight:600; }

/* ── STREAMLIT OVERRIDES ── */
.stTextArea textarea { background:rgba(255,255,255,0.03)!important; border:1px solid var(--b0)!important; border-radius:12px!important; color:var(--tx)!important; font-family:var(--font)!important; font-size:0.93rem!important; line-height:1.65!important; backdrop-filter:blur(10px)!important; resize:none!important; transition:border-color 0.2s,box-shadow 0.2s!important; }
.stTextArea textarea:focus { border-color:rgba(124,58,237,0.50)!important; box-shadow:0 0 0 3px rgba(124,58,237,0.08)!important; }
.stTextArea textarea::placeholder { color:var(--tx3)!important; }
.stTextArea label { display:none!important; }

.stButton>button { background:linear-gradient(135deg,#7c3aed,#5e18d0)!important; color:#fff!important; border:none!important; border-radius:10px!important; padding:0.65rem 1.6rem!important; font-family:var(--font)!important; font-weight:700!important; font-size:0.85rem!important; letter-spacing:0.015em!important; width:100%; box-shadow:0 4px 20px rgba(124,58,237,0.32)!important; transition:all 0.2s ease!important; }
.stButton>button:hover { background:linear-gradient(135deg,#8b5cf6,#7c3aed)!important; transform:translateY(-2px)!important; box-shadow:0 8px 32px rgba(124,58,237,0.46)!important; }
.stButton>button:active { transform:translateY(0)!important; }

[data-testid="stFileUploader"] { background:var(--g1); border:2px dashed rgba(124,58,237,0.26)!important; border-radius:14px!important; }
[data-testid="stFileUploader"]:hover { border-color:var(--purple)!important; }
[data-testid="stExpander"] { background:var(--g1)!important; border:1px solid var(--b0)!important; border-radius:12px!important; margin-bottom:5px; overflow:hidden; transition:border-color 0.2s; }
[data-testid="stExpander"]:hover { border-color:var(--bv)!important; }
[data-testid="stExpander"] summary { font-family:var(--font)!important; font-size:0.85rem!important; font-weight:500!important; color:var(--tx2)!important; }
[data-testid="stProgress"]>div>div { background:linear-gradient(90deg,#7c3aed,#a78bfa)!important; border-radius:2px!important; }
[data-testid="stProgress"]>div { background:rgba(255,255,255,0.05)!important; border-radius:2px!important; }
[data-testid="stDataFrame"] { border:1px solid var(--b0); border-radius:14px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px!important; }
hr { border-color:var(--b0)!important; }

.sfooter { text-align:center; padding:2.4rem 0; margin-top:5rem; border-top:1px solid var(--b0); color:var(--tx3); font-size:0.73rem; letter-spacing:0.05em; }
</style>""", unsafe_allow_html=True)


# =============================================================================
# CHART HELPERS
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_ml_model():
    return get_model_and_vectorizer()

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif', color='#9490b0'),
    margin=dict(l=20, r=20, t=44, b=20)
)
SC = {'Positive': '#34d399', 'Neutral': '#fbbf24', 'Negative': '#f87171'}

def make_pie_chart(pos, neu, neg):
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'], values=[pos, neu, neg], hole=0.65,
        marker=dict(colors=['#34d399', '#fbbf24', '#f87171'], line=dict(color='#05050d', width=3)),
        textfont=dict(size=13, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value} · %{percent}<extra></extra>'
    )])
    fig.update_layout(
        title=dict(text='Sentiment Distribution', font=dict(size=13, color='#edeaf8')),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(color='#9490b0')), **PT)
    return fig

def make_aspect_bar_chart(asp):
    ks = list(asp.keys())
    fig = go.Figure(data=[
        go.Bar(name='Positive', x=ks, y=[asp[a].get('Positive',0) for a in ks], marker_color='#34d399', marker_line_width=0, hovertemplate='<b>%{x}</b> — Positive: %{y}<extra></extra>'),
        go.Bar(name='Neutral',  x=ks, y=[asp[a].get('Neutral', 0) for a in ks], marker_color='#fbbf24', marker_line_width=0, hovertemplate='<b>%{x}</b> — Neutral: %{y}<extra></extra>'),
        go.Bar(name='Negative', x=ks, y=[asp[a].get('Negative',0) for a in ks], marker_color='#f87171', marker_line_width=0, hovertemplate='<b>%{x}</b> — Negative: %{y}<extra></extra>'),
    ])
    fig.update_layout(barmode='group', title=dict(text='Aspect-wise Sentiment', font=dict(size=13, color='#edeaf8')),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#9490b0')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#9490b0')),
        legend=dict(font=dict(color='#9490b0')), bargap=0.2, bargroupgap=0.1, **PT)
    return fig

def make_score_gauge(score: float):
    val   = (score + 1) * 50
    color = '#34d399' if score > 0.2 else ('#f87171' if score < -0.2 else '#fbbf24')
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
        delta={'reference': 50, 'increasing': {'color': '#34d399'}, 'decreasing': {'color': '#f87171'}},
        number={'suffix': '', 'font': {'family': 'Helvetica Neue', 'color': '#edeaf8', 'size': 30}},
        gauge={
            'axis': {'range': [0,100], 'tickcolor': '#4c4966', 'tickfont': {'color': '#4c4966'}},
            'bar': {'color': color, 'thickness': 0.2},
            'bgcolor': 'rgba(255,255,255,0.022)', 'bordercolor': 'rgba(255,255,255,0.07)',
            'steps': [
                {'range': [0,35],   'color': 'rgba(248,113,113,0.09)'},
                {'range': [35,65],  'color': 'rgba(251,191,36,0.09)'},
                {'range': [65,100], 'color': 'rgba(52,211,153,0.09)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.85, 'value': val}
        },
        title={'text': 'Sentiment Score', 'font': {'family': 'Helvetica Neue', 'color': '#9490b0', 'size': 13}}
    ))
    fig.update_layout(height=280, **PT)
    return fig


# =============================================================================
# STATE & MODEL
# =============================================================================
if "nav" not in st.session_state:
    st.session_state.nav = "landing"

with st.spinner(""):
    try:
        model, vectorizer = load_ml_model()
        model_ready = True
    except Exception as e:
        st.error(f"Model error: {e}")
        model_ready = False

nav = st.session_state.nav


# =============================================================================
# HELPERS
# =============================================================================
def topbar():
    st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">S</div>
    Sentiment<em>IQ</em>
  </div>
  <div class="topbar-status">
    <div class="topbar-dot"></div>ML &nbsp;·&nbsp; NLP
  </div>
</div>""", unsafe_allow_html=True)

def back_btn():
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.nav = "landing"
        st.rerun()


# =============================================================================
# PAGE: LANDING
# =============================================================================
if nav == "landing":
    topbar()

    st.markdown("""
<div class="hero">
  <div class="hero-orb"></div>
  <div style="position:relative;z-index:2">
    <div class="hero-tag">
      <div class="hero-tag-dot"></div>
      Aspect-Based Sentiment Analysis System For Student Feedback
    </div>
    <h1 class="hero-h1">Reveal the sentiment behind every aspect of student feedback</h1>
    <p class="hero-words"><em>Quickly</em> <em>Clearly</em> <em>Meaningfully</em></p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="strip">
  <div class="strip-item"><span class="strip-num">5</span><span class="strip-lbl">Aspects Tracked</span></div>
  <div class="strip-item"><span class="strip-num">3</span><span class="strip-lbl">Sentiment Classes</span></div>
  <div class="strip-item"><span class="strip-num">87%</span><span class="strip-lbl">Model Accuracy</span></div>
  <div class="strip-item"><span class="strip-num">571</span><span class="strip-lbl">Training Examples</span></div>
  <div class="strip-item"><span class="strip-num">&lt;1s</span><span class="strip-lbl">Inference Time</span></div>
</div>""", unsafe_allow_html=True)

    # 2.8rem gap between strip and cards
    st.markdown('<div style="margin-top:2.8rem"></div>', unsafe_allow_html=True)

    # Cards — button rendered INSIDE the visual card via fcard-foot div
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
<div class="fcard-outer">
  <div class="fcard-body">
    <div class="fcard-ico">✦</div>
    <div class="fcard-eyebrow">Single Analysis</div>
    <div class="fcard-title">Deep Sentiment Inspection</div>
    <div class="fcard-desc">Analyse one piece of feedback in full detail. Get overall sentiment, aspect breakdown, confidence scores, and probability distributions - instantly.</div>
    <div class="fcard-chips">
      <span class="chip">Faculty</span>
      <span class="chip">Infrastructure</span>
      <span class="chip">Curriculum</span>
      <span class="chip">Placements</span>
      <span class="chip">Management</span>
    </div>
  </div>
  <div class="fcard-foot">""", unsafe_allow_html=True)
        if st.button("Start Single Analysis →", key="goto_single", use_container_width=True):
            st.session_state.nav = "single"
            st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="fcard-outer">
  <div class="fcard-body">
    <div class="fcard-ico">⬆</div>
    <div class="fcard-eyebrow">Batch Analysis</div>
    <div class="fcard-title">Scale to Thousands of Rows</div>
    <div class="fcard-desc">Upload a CSV or Excel file with feedback entries. The system analyzes each row and returns a downloadable results table with visualisations.</div>
    <div class="fcard-chips">
      <span class="chip">CSV Upload</span>
      <span class="chip">XLS Upload</span>
      <span class="chip">XLSX Upload</span>
      <span class="chip">Bulk Export</span>
      <span class="chip">Visual Charts</span>
    </div>
  </div>
  <div class="fcard-foot">""", unsafe_allow_html=True)
        if st.button("Start Batch Analysis →", key="goto_batch", use_container_width=True):
            st.session_state.nav = "batch"
            st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE: SINGLE ANALYSIS
# =============================================================================
elif nav == "single" and model_ready:
    topbar()
    st.markdown('<div style="padding-top:1.4rem"></div>', unsafe_allow_html=True)
    back_btn()

    st.markdown("""
<div class="pghdr">
  <div>
    <div class="pghdr-title">Single Analysis</div>
    <div class="pghdr-sub">Enter student feedback - get aspect-level sentiment in under a second.</div>
  </div>
  <div class="pghdr-badge">✦ Deep Inspection</div>
</div>""", unsafe_allow_html=True)

    _, cx, _ = st.columns([1, 5, 1])
    with cx:
        st.markdown('<div class="izone-eyebrow">Paste or type feedback</div>', unsafe_allow_html=True)
        
        feedback_text = st.text_area(
            label="feedback",
            placeholder="e.g. The faculty explains things very clearly, but lab equipment is outdated and management never responds to student concerns…",
            height=140, label_visibility="collapsed"
        )
        bc1, right = st.columns([3, 3])

        with bc1:
            analyze_btn = st.button("Analyse Feedback →", type="primary")

        with right:
            e1, e2, e3 = st.columns(3, gap="small")

            with e1:
                ex1 = st.button("Positive", use_container_width=True)
            with e2:
                ex2 = st.button("Mixed", use_container_width=True)
            with e3:
                ex3 = st.button("Negative", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    EXAMPLES = {
        "ex1": "The professors are incredibly knowledgeable and the campus infrastructure is outstanding. Placements are great too!",
        "ex2": "Faculty teaching is excellent but the lab equipment is quite outdated. Management could be more responsive to student needs.",
        "ex3": "The curriculum is completely outdated and placements are terrible. Infrastructure needs urgent improvement."
    }
    active_text = feedback_text
    if ex1: active_text = EXAMPLES["ex1"]
    if ex2: active_text = EXAMPLES["ex2"]
    if ex3: active_text = EXAMPLES["ex3"]

    if (analyze_btn or ex1 or ex2 or ex3) and active_text.strip():
        with st.spinner("Analysing…"):
            result = analyze_feedback(active_text, model, vectorizer)

        badge_cls  = "badge-" + result['sentiment'].lower()
        score_val  = conf_to_scale(result['confidence'], result['sentiment'])
        sent_color = SC[result['sentiment']]

        atags = ""
        for a in result['aspects']:
            atags += (f'<span style="background:rgba(124,58,237,0.10);border:1px solid rgba(124,58,237,0.26);'
                      f'color:#a78bfa;padding:3px 11px;border-radius:50px;font-size:0.72rem;'
                      f'font-weight:600;margin:2px;display:inline-block;">{a}</span>')
        if not atags:
            atags = '<span style="color:#4c4966;font-size:0.82rem;">None detected</span>'

        _, rx, _ = st.columns([0.5, 5, 0.5])
        with rx:
            st.markdown(f"""
<div class="rbanner">
  <div class="rbanner-lbl">Analysis Result</div>
  <div class="rgrid">
    <div class="rtile">
      <span class="rtile-lbl">Overall Sentiment</span>
      <span class="sentiment-badge {badge_cls}" style="font-size:0.82rem;">{result['emoji']} {result['sentiment']}</span>
    </div>
    <div class="rtile">
      <span class="rtile-lbl">Confidence Score</span>
      <span class="rtile-val" style="color:{sent_color}">{score_val} / 10</span>
    </div>
    <div class="rtile">
      <span class="rtile-lbl">Sentiment Score</span>
      <span class="rtile-val" style="color:{sent_color}">{'+' if result['overall_score']>0 else ''}{result['overall_score']:.2f}</span>
    </div>
    <div class="rtile">
      <span class="rtile-lbl">Detected Aspects</span>
      <div style="display:flex;flex-wrap:wrap;gap:3px;justify-content:center;margin-top:4px">{atags}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        lc, rc = st.columns(2, gap="medium")
        with lc:
            
            st.markdown('<div class="ipanel-lbl">Aspect-wise Breakdown</div>', unsafe_allow_html=True)
            for ar in result["aspect_results"]:
                bc_ = f"sentiment-badge badge-{ar['sentiment'].lower()}"
                bar = SC[ar['sentiment']]
                pct = ar['confidence'] * 100
                st.markdown(f"""
<div class="acard">
  <div style="flex:1">
    <div class="acard-name">{ar['aspect']}</div>
    <div class="acard-score">{conf_to_scale(ar['confidence'],ar['sentiment'])}/10 · {ar['score']:+.1f}</div>
    <div class="abar-wrap"><div class="abar" style="width:{pct}%;background:{bar}"></div></div>
  </div>
  <span class="{bc_}" style="margin-left:1rem;flex-shrink:0">{ar['emoji']} {ar['sentiment']}</span>
</div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rc:
            st.markdown('<div class="ipanel-lbl">Probability Distribution</div>', unsafe_allow_html=True)
            proba = result["probabilities"]
            if proba:
                lbls = list(proba.keys())
                vals = [v * 100 for v in proba.values()]
                fig_p = go.Figure(go.Bar(
                    x=lbls, y=vals,
                    marker=dict(color=[SC.get(l,'#a78bfa') for l in lbls], line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in vals], textposition='outside',
                    textfont=dict(color='#9490b0'),
                    hovertemplate='<b>%{x}</b>: %{y:.1f}%<extra></extra>'
                ))
                fig_p.update_layout(
                    yaxis=dict(range=[0,118], gridcolor='rgba(255,255,255,0.05)',
                               ticksuffix='%', tickfont=dict(color='#9490b0')),
                    xaxis=dict(tickfont=dict(color='#9490b0')),
                    height=280, **PT)
                st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

       


# =============================================================================
# PAGE: BATCH ANALYSIS
# =============================================================================
elif nav == "batch" and model_ready:
    topbar()
    st.markdown('<div style="padding-top:1.4rem"></div>', unsafe_allow_html=True)
    back_btn()

    st.markdown("""
<div class="pghdr">
  <div>
    <div class="pghdr-title">Batch Analysis</div>
    <div class="pghdr-sub">Upload a dataset and analyse all feedback at once — results ready in seconds.</div>
  </div>
  <div class="pghdr-badge">⬆ Bulk Processing</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="bgrid">
  <div class="bpanel">
    <div class="bpanel-lbl">How It Works</div>
    <div style="font-size:0.86rem;color:#9490b0;line-height:1.9">
      <div style="margin-bottom:0.5rem"><span style="color:#a78bfa;font-weight:700">01&nbsp;&nbsp;</span>Upload a CSV or Excel file</div>
      <div style="margin-bottom:0.5rem"><span style="color:#a78bfa;font-weight:700">02&nbsp;&nbsp;</span>Ensure one column contains feedback text</div>
      <div style="margin-bottom:0.5rem"><span style="color:#a78bfa;font-weight:700">03&nbsp;&nbsp;</span>Click Run Batch Analysis</div>
      <div><span style="color:#a78bfa;font-weight:700">04&nbsp;&nbsp;</span>Download annotated results as CSV</div>
    </div>
  </div>
  <div class="bpanel">
    <div class="bpanel-lbl">Accepted Column Names</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px">
      <span class="chip">feedback</span><span class="chip">response</span>
      <span class="chip">review</span><span class="chip">comment</span>
      <span class="chip">opinion</span><span class="chip">text</span>
      <span class="chip">remark</span><span class="chip">note</span>
      <span class="chip">report</span><span class="chip">suggestion</span>
    </div>
    <div style="margin-top:1rem;font-size:0.81rem;color:#4c4966">Column names are case-insensitive.</div>
  </div>
</div>""", unsafe_allow_html=True)

    uc, dc = st.columns([3, 1])
    with uc:
        uploaded_file = st.file_uploader("Upload file", type=["csv","xls","xlsx"], label_visibility="collapsed")
    with dc:
        sample_csv = pd.DataFrame({'feedback': [
            "The faculty explains concepts very well.",
            "Lab equipment is outdated and wifi is slow.",
            "Placements are decent but could improve.",
            "Management is unresponsive to complaints.",
            "The curriculum is well structured.",
        ]})
        st.download_button("⬇ Sample CSV", sample_csv.to_csv(index=False), file_name="sample_feedback.csv", mime="text/csv")

    if uploaded_file:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith('.csv'):    df = pd.read_csv(uploaded_file)
            elif fname.endswith('.xls'):  df = pd.read_excel(uploaded_file, engine='xlrd')
            elif fname.endswith('.xlsx'): df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:                         df = pd.read_csv(uploaded_file)

            ACCEPTED_COLUMNS = ['feedback','feedbacks','report','reports','response','responses',
                'opinion','opinions','review','reviews','judgement','judgements','judgment','judgments',
                'note','notes','comment','comments','text','texts','remark','remarks',
                'suggestion','suggestions','observation','observations','entry','entries',
                'input','inputs','answer','answers','student_feedback','student_response','student_comment']
            col_match = next((c for c in df.columns if c.strip().lower().replace(" ","_") in ACCEPTED_COLUMNS), None)
            if col_match:
                df = df.rename(columns={col_match: 'feedback'})

            if 'feedback' not in df.columns:
                st.error("Column not found. Rename to: feedback / response / review / comment / text")
            else:
                df = df.dropna(subset=['feedback'])
                st.success(f"✓ Loaded {len(df)} feedback entries")

                if st.button("Run Batch Analysis →", type="primary"):
                    prog = st.progress(0); status = st.empty()
                    texts = df['feedback'].tolist(); batch = []
                    for i, text in enumerate(texts):
                        r = analyze_feedback(str(text), model, vectorizer)
                        batch.append(r)
                        prog.progress((i+1)/len(texts))
                        status.text(f"Analysing {i+1}/{len(texts)}…")
                    prog.empty(); status.empty()
                    st.session_state.batch_results = batch
                    st.session_state.show_count    = 10

                if 'batch_results' in st.session_state and st.session_state.batch_results:
                    results = st.session_state.batch_results
                    show_n  = st.session_state.get('show_count', 10)
                    stats   = compute_summary_stats(results)

                    st.markdown('<div class="slbl">Overview</div>', unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    for col, lbl, cnt, cls in [
                        (m1,"Total",str(stats['total']),"total"),
                        (m2,"Positive",str(stats['positive']),"positive"),
                        (m3,"Neutral",str(stats['neutral']),"neutral"),
                        (m4,"Negative",str(stats['negative']),"negative"),
                    ]:
                        with col:
                            st.markdown(f'<div class="mc {cls}"><div class="mc-val">{cnt}</div><div class="mc-lbl">{lbl}</div></div>', unsafe_allow_html=True)

                    st.markdown('<div class="slbl">Visualisations</div>', unsafe_allow_html=True)
                    ch1, ch2 = st.columns([1, 2])
                    with ch1:
                        st.markdown('<div class="cpanel">', unsafe_allow_html=True)
                        st.plotly_chart(make_pie_chart(stats['positive'],stats['neutral'],stats['negative']), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with ch2:
                        if stats['aspect_sentiment']:
                            st.markdown('<div class="cpanel">', unsafe_allow_html=True)
                            st.plotly_chart(make_aspect_bar_chart(stats['aspect_sentiment']), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="slbl">Results Table</div>', unsafe_allow_html=True)
                    result_df = pd.DataFrame([{
                        'Feedback':   r['original_text'],
                        'Aspects':    ', '.join(r['aspects']),
                        'Sentiment':  r['sentiment'],
                        'Confidence': f"{conf_to_scale(r['confidence'],r['sentiment'])}/10",
                        'Score':      r['overall_score']
                    } for r in results])

                    def color_sentiment(val):
                        c = {'Positive':'rgba(52,211,153,0.08)','Negative':'rgba(248,113,113,0.08)','Neutral':'rgba(251,191,36,0.08)'}
                        return f'background-color:{c.get(val,"")};color:white'

                    styled_df = result_df.style.applymap(color_sentiment, subset=['Sentiment'])
                    st.dataframe(styled_df, use_container_width=True, height=360,
                        column_config={
                            "Feedback":   st.column_config.TextColumn("Feedback",   width="large"),
                            "Aspects":    st.column_config.TextColumn("Aspects",    width="medium"),
                            "Sentiment":  st.column_config.TextColumn("Sentiment",  width="small"),
                            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                            "Score":      st.column_config.NumberColumn("Score",    width="small", format="%.2f"),
                        })
                    _, dl = st.columns([3,1])
                    with dl:
                        st.download_button("⬇ Download CSV", result_df.to_csv(index=False), file_name="absa_results.csv", mime="text/csv")

                    st.markdown('<div class="slbl">Full Feedback Details</div>', unsafe_allow_html=True)
                    for i, r in enumerate(results[:show_n]):
                        sent = r['sentiment']; clr = SC[sent]
                        with st.expander(f"{r['emoji']}  #{i+1}  —  {r['original_text'][:90]}…"):
                            st.markdown(f'<div style="background:rgba(255,255,255,0.022);border-left:2px solid {clr};border-radius:10px;padding:1rem 1.4rem;color:#edeaf8;font-size:0.9rem;line-height:1.75">' + r['original_text'] + '</div>', unsafe_allow_html=True)
                            cc1,cc2,cc3 = st.columns(3)
                            with cc1:
                                st.markdown(f'<div style="color:#4c4966;font-size:0.61rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px">Sentiment</div><div style="color:{clr};font-size:0.9rem;font-weight:700">{r["emoji"]} {sent}</div>', unsafe_allow_html=True)
                            with cc2:
                                st.markdown(f'<div style="color:#4c4966;font-size:0.61rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px">Aspects</div><div style="color:#a78bfa;font-size:0.9rem;font-weight:600">{", ".join(r["aspects"])}</div>', unsafe_allow_html=True)
                            with cc3:
                                st.markdown(f'<div style="color:#4c4966;font-size:0.61rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:4px">Score</div><div style="color:#edeaf8;font-size:0.9rem;font-weight:700">{conf_to_scale(r["confidence"],r["sentiment"])}/10</div>', unsafe_allow_html=True)

                    total_n = len(results)
                    if show_n < total_n:
                        _, pb, _ = st.columns([2,1,2])
                        with pb:
                            if st.button(f"Load {total_n-show_n} More", key="see_more"):
                                st.session_state.show_count = show_n + 10; st.rerun()
                    elif total_n > 10:
                        _, pb, _ = st.columns([2,1,2])
                        with pb:
                            if st.button("Show Less", key="show_less"):
                                st.session_state.show_count = 10; st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")


# =============================================================================
# ABOUT — expander at bottom of every page
# =============================================================================
st.markdown('<div style="margin-top:4rem"></div>', unsafe_allow_html=True)
with st.expander("About SentimentIQ"):
    st.markdown("""
<div class="agrid">
  <div class="apanel">
    <h3>What It Does</h3>
    <p><b>Aspect-Based Sentiment Analysis on Student Feedback</b> goes beyond simple positive/negative labelling. It first detects <b>which aspect</b> of the institution the student is discussing, then classifies the sentiment expressed about that specific aspect.</p>
    <br>
    <p>This allows institutions to pinpoint exactly where students are satisfied or dissatisfied — whether it's faculty quality, infrastructure, curriculum relevance, placement outcomes, or administrative management.</p>
  </div>
  <div class="apanel">
    <h3>Pipeline</h3>
    <div style="font-size:0.86rem;color:#9490b0;line-height:2">
      <div><span style="color:#a78bfa;font-weight:700">1. Preprocessing</span></div>
      <div style="color:#4c4966;padding-left:1rem;margin-bottom:0.6rem">Lowercase → Strip punctuation → Tokenize → Remove stopwords → Lemmatize</div>
      <div><span style="color:#a78bfa;font-weight:700">2. Aspect Extraction</span></div>
      <div style="color:#4c4966;padding-left:1rem;margin-bottom:0.6rem">Keyword matching against curated per-aspect dictionaries</div>
      <div><span style="color:#a78bfa;font-weight:700">3. Classification</span></div>
      <div style="color:#4c4966;padding-left:1rem">TF-IDF (unigrams + bigrams, 5k features) → Logistic Regression</div>
    </div>
  </div>
</div>
<div class="agrid" style="margin-top:0">
  <div class="apanel">
    <h3>Tracked Aspects</h3>
    <ul>
      <li><b>Faculty</b> — Teaching quality, availability, expertise</li>
      <li><b>Infrastructure</b> — Labs, wifi, library, campus</li>
      <li><b>Curriculum</b> — Syllabus relevance, course structure</li>
      <li><b>Placements</b> — Job offers, companies, packages</li>
      <li><b>Management</b> — Administration, responsiveness</li>
    </ul>
  </div>
  <div class="apanel">
    <h3>Model Specifications</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem">
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;text-align:center">
        <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-0.04em">87%</div>
        <div style="font-size:0.62rem;font-weight:700;color:#4c4966;letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">Accuracy</div>
      </div>
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;text-align:center">
        <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-0.04em">571</div>
        <div style="font-size:0.62rem;font-weight:700;color:#4c4966;letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">Training Examples</div>
      </div>
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;text-align:center">
        <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-0.04em">5k</div>
        <div style="font-size:0.62rem;font-weight:700;color:#4c4966;letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">TF-IDF Features</div>
      </div>
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;text-align:center">
        <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-0.04em">3</div>
        <div style="font-size:0.62rem;font-weight:700;color:#4c4966;letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">Sentiment Classes</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="sfooter">
  SentimentIQ &nbsp;·&nbsp; Aspect-Based Sentiment Analysis on Student Feedback
  &nbsp;·&nbsp; Powered by NLP + Logistic Regression
</div>""", unsafe_allow_html=True)
