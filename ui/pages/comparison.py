"""Page: compare selected tracked players."""
from __future__ import annotations

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from modules import heatmap_generator as hg
from modules import movement_analyzer as ma
from ui import components, state

MAX_PLAYERS = 4


def render() -> None:
    components.page_header("Player comparison", "Compare two to four tracked IDs on the measures the system can measure reliably.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    data = state.bundle(run)
    metrics, steps = data["metrics"], data["steps"]
    stable = state.stable_ids(metrics)
    if len(stable) < 2:
        st.info("At least two stable tracks are needed for a comparison.")
        return
    u = components.units(run)
    components.unit_note(run)

    chosen = st.multiselect(f"Tracks to compare (2 to {MAX_PLAYERS})", stable, default=stable[:2],
                            max_selections=MAX_PLAYERS, format_func=lambda i: f"ID {i}")
    if len(chosen) < 2:
        st.info("Select at least two tracks.")
        return
    colours = components.series_colours(chosen, "comparison")
    rows = metrics.set_index("track_id").loc[chosen]
    spans = rows["visible_s"]
    if spans.max() > 1.5 * spans.min():
        st.warning(f"The selected IDs were tracked for different lengths of time ({spans.min():.1f} s to "
                   f"{spans.max():.1f} s). Compare averages and peaks rather than totals such as distance.")
    if not run.is_calibrated:
        st.warning("Pixel speeds depend on the distance from the camera, so they only compare fairly for players in "
                   "similar parts of the pitch.")

    st.header("Measures")
    measures = [("visible_s", "Tracked time (s)"), ("distance", f"Distance ({u['dist']})"),
                ("avg_speed", f"Average speed ({u['speed']})"), ("peak_speed", f"Peak speed ({u['speed']})"),
                ("area_covered", f"Area covered ({u['area']})")]
    if run.is_calibrated:
        measures.append(("sprints", f"Sprints above {config.SPRINT_THRESHOLD_MPS} m/s"))
    table = pd.DataFrame({f"ID {i}": [components.fmt(rows.loc[i, key], 0 if key == "sprints" else 2)
                                      for key, _ in measures] for i in chosen},
                         index=[label for _, label in measures])
    table.loc["Colour group"] = [rows.loc[i, "colour_group"] for i in chosen]
    table.loc["Tracked between (s)"] = [f"{rows.loc[i, 'first_seen_s']:.1f} to {rows.loc[i, 'last_seen_s']:.1f}"
                                        for i in chosen]
    st.dataframe(table, width="stretch")

    fig = make_subplots(rows=1, cols=len(measures), subplot_titles=[label for _, label in measures],
                        horizontal_spacing=0.06)
    for col, (key, _) in enumerate(measures, start=1):
        values = [rows.loc[i, key] for i in chosen]
        fig.add_trace(go.Bar(
            x=[f"ID {i}" for i in chosen], y=values, marker=dict(color=[colours[i] for i in chosen]),
            text=[components.fmt(v, 0 if key == "sprints" else 1) for v in values], textposition="outside",
            textfont=dict(color=hg.INK, size=11), cliponaxis=False, showlegend=False,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ), row=1, col=col)
    hg.base_layout(fig, None, 340)
    fig.update_layout(bargap=0.3, margin=dict(t=60))
    fig.update_annotations(font=dict(size=11, color=hg.INK_SECONDARY))
    fig.update_yaxes(rangemode="tozero", showticklabels=False)
    components.plot(fig, "compare_bars")
    st.caption("Each small chart has its own scale. Bar colours follow the IDs in the selection above.")

    st.header("Speed over time")
    fig = go.Figure()
    for i in chosen:
        series = ma.speed_series(steps, i, fps=run.effective_fps)
        if not series.empty:
            fig.add_trace(go.Scatter(x=series["time_s"], y=series["speed"], mode="lines", name=f"ID {i}",
                                     line=dict(color=colours[i], width=2),
                                     hovertemplate=f"ID {i}: " + "%{y:.2f} at %{x:.1f} s<extra></extra>"))
    hg.base_layout(fig, None, 320)
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title=f"speed ({u['speed']}), 1 s rolling mean", rangemode="tozero")
    components.plot(fig, "compare_speed")

    st.header("Movement paths")
    paths = {f"ID {i}": steps[steps["track_id"] == i].sort_values("time_s") for i in chosen}
    labelled = {f"ID {i}": colours[i] for i in chosen}
    if run.is_calibrated:
        cal = run.calibration
        components.plot(hg.pitch_paths(paths, labelled, cal.pitch_length, cal.pitch_width), "compare_paths")
    else:
        reference = cv2.cvtColor(cv2.imread(str(run.reference_frame)), cv2.COLOR_BGR2RGB)
        components.plot(hg.image_paths(reference, paths, labelled), "compare_paths")
