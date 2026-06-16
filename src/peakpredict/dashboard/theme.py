"""Visual theme for the dashboard: fonts, CSS variables, component styling, and
the branded header — all injected once so the page modules stay logic-only.

Aesthetic: a precision-instrument / track-timing look. Warm paper, near-black
ink, a single hot vermilion accent, a dark instrument-panel sidebar. Display
type is Bricolage Grotesque; body is IBM Plex Sans; data/labels are IBM Plex
Mono (a stopwatch/scoreboard cue for athletics).
"""

from __future__ import annotations

import streamlit as st

# Chart colours are exported so charting.py shares the exact palette.
INK = "#14161B"
INK_SOFT = "#60656F"
PAPER = "#F4F2EC"
SURFACE = "#FFFFFF"
LINE = "#E5E0D6"
ACCENT = "#FF4D17"
ATHLETE = "#16263A"
REFERENCE = "#B7B0A2"
FONT_SANS = "IBM Plex Sans, system-ui, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"
FONT_DISPLAY = "Bricolage Grotesque, " + FONT_SANS

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --paper:#F4F2EC; --surface:#FFFFFF; --ink:#14161B; --ink-soft:#60656F;
  --line:#E5E0D6; --accent:#FF4D17; --accent-soft:rgba(255,77,23,.12);
  --athlete:#16263A; --reference:#B7B0A2;
  --sans:'IBM Plex Sans',system-ui,sans-serif; --mono:'IBM Plex Mono',monospace;
  --display:'Bricolage Grotesque',var(--sans);
}
.stApp{background:var(--paper);}
html,body,[data-testid="stAppViewContainer"],.stMarkdown,p,div,span,label,input,button,select{font-family:var(--sans);}
/* let our header clear Streamlit's floating top toolbar */
[data-testid="stHeader"]{background:rgba(0,0,0,0);}
.block-container{max-width:1280px;padding-top:3.6rem;padding-bottom:3rem;}
h1,h2,h3,h4{font-family:var(--display);letter-spacing:-.022em;color:var(--ink);font-weight:700;}
h1{font-size:2rem;margin-bottom:.15rem;} h2{font-size:1.4rem;} h3{font-size:1.05rem;}

/* eyebrow / section label */
.pp-eyebrow{font-family:var(--mono);text-transform:uppercase;letter-spacing:.18em;
  font-size:.66rem;color:var(--ink-soft);margin:.6rem 0 .35rem;}

/* branded header bar */
.pp-header{display:flex;align-items:baseline;gap:1rem;border-bottom:2px solid var(--ink);
  padding:.1rem 0 .7rem;margin:.2rem 0 1.3rem;}
.pp-mark{font-family:var(--display);font-weight:800;font-size:1.5rem;letter-spacing:-.03em;
  color:var(--ink);line-height:1;}
.pp-mark b{color:var(--accent);font-weight:800;}
.pp-tag{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.14em;
  color:var(--ink-soft);}
.pp-ver{margin-left:auto;font-family:var(--mono);font-size:.68rem;color:var(--ink);
  background:var(--accent-soft);border:1px solid var(--accent);padding:.24rem .6rem;
  border-radius:999px;letter-spacing:.05em;}

/* sidebar -> dark instrument panel */
[data-testid="stSidebar"]{background:var(--ink);border-right:1px solid #000;}
[data-testid="stSidebar"] *{color:#E9E6DF;}
[data-testid="stSidebar"] .pp-mark{color:#fff;font-size:1.25rem;}
[data-testid="stSidebar"] .pp-eyebrow{color:#9A968D;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.1rem;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.18rem .1rem;font-weight:500;}

/* metric cards */
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;box-shadow:0 1px 2px rgba(20,22,27,.04);}
[data-testid="stMetricValue"]{font-family:var(--mono);font-weight:600;color:var(--ink);font-size:1.5rem;}
[data-testid="stMetricLabel"]{font-family:var(--mono);text-transform:uppercase;letter-spacing:.1em;
  font-size:.64rem;color:var(--ink-soft);}

/* inputs */
[data-baseweb="select"]>div,.stTextInput input{
  border-radius:8px!important;border-color:var(--line)!important;}
[data-baseweb="select"] *{font-family:var(--sans);}

/* buttons -> ink, accent on hover */
.stButton>button{background:var(--ink);color:#fff;border:none;border-radius:8px;font-weight:600;
  letter-spacing:.02em;padding:.5rem 1.15rem;transition:transform .08s ease,background .15s ease;}
.stButton>button:hover{background:var(--accent);transform:translateY(-1px);color:#fff;}

/* dataframe + captions */
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stCaptionContainer"],.stCaption,[data-testid="stCaptionContainer"] p{
  font-family:var(--mono);letter-spacing:.03em;color:var(--ink-soft);font-size:.74rem;}
hr{border-color:var(--line);}
</style>
"""


def inject() -> None:
    """Inject fonts + CSS. Call once at the top of the app."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header(version: str) -> None:
    """Render the slim context bar (the sidebar carries the wordmark)."""
    st.markdown(
        '<div class="pp-header">'
        '<span class="pp-tag">athletics peak-performance projection</span>'
        f'<span class="pp-ver">DATA · {version}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> None:
    """Render a small monospace section label."""
    st.markdown(f'<div class="pp-eyebrow">{text}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Render the wordmark in the sidebar."""
    st.sidebar.markdown('<div class="pp-mark">PEAK<b>PREDICTOR</b></div>', unsafe_allow_html=True)
