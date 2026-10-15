"""Football Video Analysis System: Streamlit entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import config

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=str(config.ASSETS_DIR / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import state, theme  # noqa: E402
from ui.pages import (  # noqa: E402
    about, analyse, calibration, coach, comparison, heatmaps, legal, overview, pose, reports, statistics, tracking,
)

theme.apply()

pages = {
    "analyse": st.Page(analyse.render, title="Analyse a video", url_path="analyse", default=True),
    "overview": st.Page(overview.render, title="Overview", url_path="overview"),
    "tracking": st.Page(tracking.render, title="Player tracking", url_path="tracking"),
    "heatmaps": st.Page(heatmaps.render, title="Heatmaps and movement", url_path="heatmaps"),
    "statistics": st.Page(statistics.render, title="Performance statistics", url_path="statistics"),
    "comparison": st.Page(comparison.render, title="Player comparison", url_path="comparison"),
    "coach": st.Page(coach.render, title="AI coach", url_path="coach"),
    "reports": st.Page(reports.render, title="Reports and export", url_path="reports"),
    "calibration": st.Page(calibration.render, title="Pitch calibration", url_path="calibration"),
    "pose": st.Page(pose.render, title="Pose estimation (optional)", url_path="pose"),
    "about": st.Page(about.render, title="About and method", url_path="about"),
    "privacy": st.Page(legal.render_privacy, title="Privacy policy", url_path="privacy"),
    "terms": st.Page(legal.render_terms, title="Terms of use", url_path="terms"),
}
st.session_state["pages"] = pages

st.logo(str(config.ASSETS_DIR / "wordmark.png"), size="large", icon_image=str(config.ASSETS_DIR / "favicon.png"))

navigation = st.navigation({
    "Analysis": [pages[k] for k in ("analyse", "overview", "tracking", "heatmaps", "statistics", "comparison",
                                    "coach", "reports")],
    "Tools": [pages["calibration"], pages["pose"]],
    "Information": [pages["about"], pages["privacy"], pages["terms"]],
}, expanded=True)

with st.sidebar:
    available = state.runs()
    if available:
        labels = {r["run_dir"]: f"{r['video']}  ({r['created'][:16]})" for r in available}
        current = state.active_run_dir()
        options = list(labels)
        chosen = st.selectbox("Active analysis", options, index=options.index(current) if current in options else 0,
                              format_func=lambda d: labels[d],
                              help="Every page shows results for this analysis.")
        if chosen != current:
            state.set_active_run(chosen)
            st.rerun()
    else:
        st.caption("No saved analyses yet.")
    st.caption(f"Version {config.APP_VERSION}. Runs locally; videos never leave this computer.")

navigation.run()
