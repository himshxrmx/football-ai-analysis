"""Page styling that the Streamlit theme options in .streamlit/config.toml cannot express."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Remove the coloured decoration strip and the hosted-app badge. */
[data-testid="stDecoration"], .viewerBadge_container__1QSob, footer { display: none !important; }

/* Tighter top spacing for a dashboard layout. */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

h1, h2, h3 { letter-spacing: -0.01em; }
h1 { font-size: 1.75rem !important; font-weight: 650 !important; }
h2 { font-size: 1.3rem !important; font-weight: 620 !important; margin-top: 0.6rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

/* Metric cards: flat panels with a hairline border. */
[data-testid="stMetric"] { background: #ffffff; }
[data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #52514e; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 620; }

/* Sidebar brand line. */
.app-brand { display: flex; align-items: center; gap: 10px; margin: 0 0 4px 0; }
.app-brand img { width: 28px; height: 28px; }
.app-brand span { font-weight: 650; font-size: 1.02rem; color: #e6ebe5; }
.app-sub { color: #a9b6ac; font-size: 0.8rem; margin: 0 0 6px 0; }

/* Insight cards. */
.insight-kind { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #52514e; margin-bottom: 2px; }
.note { color: #52514e; font-size: 0.88rem; }
.swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid #888; vertical-align: middle; margin-right: 6px; }
</style>
"""


def apply() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
