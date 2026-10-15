"""Page: PDF report and CSV / video downloads."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import coaching_insights as ci
from modules.report_generator import build_pdf_report
from ui import components, state


def render() -> None:
    components.page_header("Reports and export", "Download the measured data, the annotated video and a PDF summary of the active analysis.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    data = state.bundle(run)
    metrics, steps, zones, overview, spread = (data["metrics"], data["steps"], data["zones"], data["overview"],
                                               data["spread"])
    space = "pitch" if run.is_calibrated else "image"

    st.header("PDF report")
    st.write("The report contains the video and processing details, summary statistics, charts, the player "
             "statistics table, the coaching insights for the three longest-tracked IDs and the method's limitations.")
    pdf_path = run.run_dir / "report.pdf"
    if st.button("Build PDF report", type="primary"):
        team = ci.team_insights(metrics, spread, overview, space)
        top = state.stable_ids(metrics)[:3]
        players = {i: ci.player_insights(i, metrics, zones, space, overview["duration_processed_s"]) for i in top}
        try:
            with st.spinner("Building the report..."):
                build_pdf_report(run, metrics, overview, steps, team, players, pdf_path)
            st.session_state["report_built"] = str(pdf_path)
        except Exception as exc:
            st.error(f"The report could not be created: {exc}")
    if pdf_path.exists():
        st.download_button("Download PDF report", pdf_path.read_bytes(), f"{run.run_id}_report.pdf", "application/pdf")
        if st.session_state.get("report_built") == str(pdf_path):
            st.success(f"Report saved to outputs/{run.run_id}/report.pdf")

    st.header("Data files (CSV)")
    files = [
        ("Tracking data", "One row per tracked person per analysed frame: time, ID, box, confidence, feet position "
                          "and reference-view position.", run.run_dir / "tracks.csv", "tracks.csv"),
        ("Per-frame summary", "People tracked, detections, ball detection and camera motion per analysed frame.",
         run.run_dir / "frames.csv", "frames.csv"),
        ("Ball detections", "The most confident ball detection in each frame where one was found.",
         run.run_dir / "ball.csv", "ball.csv"),
    ]
    for title, description, path, name in files:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{title}**  \n{description}")
        if path.exists():
            c2.download_button(f"Download {name}", path.read_bytes(), f"{run.run_id}_{name}", "text/csv", key=name)
        else:
            c2.caption("Not available")

    derived = [
        ("Player statistics", "One row per track ID with the movement measures shown in the dashboard.",
         metrics.to_csv(index=False), "player_statistics.csv"),
        ("Zone shares", "Share of positions per third and channel for every track.", zones.to_csv(index=False),
         "zones.csv"),
    ]
    team = ci.team_insights(metrics, spread, overview, space)
    frames = [ci.insights_dataframe(team, "whole clip")]
    for i in state.stable_ids(metrics):
        frames.append(ci.insights_dataframe(ci.player_insights(i, metrics, zones, space, overview["duration_processed_s"]),
                                            f"ID {i}"))
    derived.append(("Coaching insights", "Every insight for the clip and for each stable track, with its rule and "
                                         "evidence.", pd.concat(frames, ignore_index=True).to_csv(index=False),
                    "insights.csv"))
    for title, description, text, name in derived:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{title}**  \n{description}")
        c2.download_button(f"Download {name}", text.encode("utf-8"), f"{run.run_id}_{name}", "text/csv", key=name)

    st.header("Video")
    if run.annotated_video.exists():
        size_mb = run.annotated_video.stat().st_size / (1024 * 1024)
        st.download_button(f"Download annotated video ({size_mb:.1f} MB)", run.annotated_video.read_bytes(),
                           f"{run.run_id}_annotated.mp4", "video/mp4")
    st.caption("Charts on every page can also be saved as PNG with the camera button in their top-right corner.")
