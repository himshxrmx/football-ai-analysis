"""Page: per-player performance statistics."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import config
from modules import heatmap_generator as hg
from modules import movement_analyzer as ma
from ui import components, state


def render() -> None:
    components.page_header("Performance statistics", "Movement measured for every tracked ID. Only values the system actually measures are shown.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    data = state.bundle(run)
    metrics, steps, zones = data["metrics"], data["steps"], data["zones"]
    if metrics.empty:
        st.info("No tracks were recorded for this analysis.")
        return
    u = components.units(run)
    components.unit_note(run)

    c1, c2, c3 = st.columns(3)
    longest = float(max(metrics["visible_s"].max(), 1.0))
    min_visible = c1.slider("Minimum tracked time (s)", 0.0, longest, min(config.MIN_TRACK_SECONDS, longest), 0.5)
    groups = sorted(metrics["colour_group"].unique())
    group_filter = c2.multiselect("Colour groups", groups, default=groups)
    roles = sorted(metrics["role"].unique())
    role_filter = c3.multiselect("Detected class", roles, default=roles,
                                 help="With the COCO model every track is 'person'. A football model adds player, "
                                      "goalkeeper and referee.")
    table = metrics[(metrics["visible_s"] >= min_visible) & metrics["colour_group"].isin(group_filter)
                    & metrics["role"].isin(role_filter)].copy()
    st.caption(f"{len(table)} of {len(metrics)} track IDs match the filters.")
    if table.empty:
        st.info("No tracks match the filters.")
        return

    st.header("Statistics table")
    columns = {
        "track_id": st.column_config.NumberColumn("ID", format="%d"),
        "colour_group": "Colour group",
        "visible_s": st.column_config.NumberColumn("Tracked (s)", format="%.1f"),
        "visibility_pct": st.column_config.NumberColumn("Of clip (%)", format="%.0f"),
        "distance": st.column_config.NumberColumn(f"Distance ({u['dist']})", format="%.1f"),
        "avg_speed": st.column_config.NumberColumn(f"Avg speed ({u['speed']})", format="%.2f"),
        "peak_speed": st.column_config.NumberColumn(f"Peak speed ({u['speed']})", format="%.2f",
                                                    help="95th percentile of smoothed speed."),
        "area_covered": st.column_config.NumberColumn(f"Area covered ({u['area']})", format="%.0f",
                                                      help="Area of the convex hull around all positions."),
        "first_seen_s": st.column_config.NumberColumn("First seen (s)", format="%.1f"),
        "last_seen_s": st.column_config.NumberColumn("Last seen (s)", format="%.1f"),
        "mean_confidence": st.column_config.NumberColumn("Mean confidence", format="%.2f"),
    }
    order = list(columns)
    if run.is_calibrated:
        columns["high_speed_distance"] = st.column_config.NumberColumn(
            f"Distance above {config.HIGH_SPEED_THRESHOLD_MPS} m/s (m)", format="%.1f")
        columns["sprints"] = st.column_config.NumberColumn(f"Sprints above {config.SPRINT_THRESHOLD_MPS} m/s", format="%d")
        order[7:7] = ["high_speed_distance", "sprints"]
    st.dataframe(table.sort_values("visible_s", ascending=False), hide_index=True, width="stretch",
                 column_order=order, column_config=columns)
    st.download_button("Download statistics (CSV)", table.to_csv(index=False).encode("utf-8"),
                       f"{run.run_id}_player_statistics.csv", "text/csv")

    st.header("Compare one measure across tracks")
    options = {"distance": f"Distance ({u['dist']})", "avg_speed": f"Average speed ({u['speed']})",
               "peak_speed": f"Peak speed ({u['speed']})", "visible_s": "Tracked time (s)",
               "area_covered": f"Area covered ({u['area']})"}
    if run.is_calibrated:
        options["high_speed_distance"] = f"Distance above {config.HIGH_SPEED_THRESHOLD_MPS} m/s (m)"
    measure = st.selectbox("Measure", list(options), format_func=options.get)
    ranked = table.dropna(subset=[measure]).sort_values(measure, ascending=False).head(30)
    fig = go.Figure(go.Bar(
        x=[f"ID {i}" for i in ranked["track_id"]], y=ranked[measure], marker=dict(color=hg.SERIES[0]),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    hg.base_layout(fig, None, 360)
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(title=options[measure], rangemode="tozero")
    components.plot(fig, "measure_bar")
    st.caption("Tracks that were visible for longer naturally collect more distance. Compare speeds, or filter by "
               "tracked time, for a fairer comparison." + ("" if run.is_calibrated else
               " In pixel space, players far from the camera move fewer pixels for the same distance."))

    st.header("Speed over time")
    ids = table.sort_values("visible_s", ascending=False)["track_id"].astype(int).tolist()
    player = st.selectbox("Track", ids, format_func=lambda i: f"ID {i}", key="speed_player")
    series = ma.speed_series(steps, player, fps=run.effective_fps)
    if series.empty:
        st.info("Not enough consecutive positions to compute a speed for this track.")
    else:
        fig = go.Figure(go.Scatter(x=series["time_s"], y=series["speed"], mode="lines",
                                   line=dict(color=hg.SERIES[0], width=2),
                                   hovertemplate="%{x:.1f} s: %{y:.2f}<extra></extra>"))
        hg.base_layout(fig, None, 300)
        fig.update_xaxes(title="time (s)")
        fig.update_yaxes(title=f"speed ({u['speed']}), 1 s rolling mean", rangemode="tozero")
        if run.is_calibrated:
            fig.add_hline(y=config.HIGH_SPEED_THRESHOLD_MPS, line=dict(color=hg.MUTED, width=1, dash="dot"),
                          annotation_text="high-speed threshold", annotation_font_color=hg.INK_SECONDARY)
        components.plot(fig, "speed_series")

    st.header("Zones")
    z = zones[zones["track_id"].isin(table["track_id"])].copy()
    frame_of_reference = "pitch" if run.is_calibrated else "camera view (rough indication only)"
    st.caption(f"Share of tracked positions in each third and channel of the {frame_of_reference}. Left and right "
               "are as seen from the main camera; far is the touchline away from the camera.")
    st.dataframe(z.rename(columns={"track_id": "ID", "left_third": "Left third %", "middle_third": "Middle third %",
                                   "right_third": "Right third %", "far_channel": "Far channel %",
                                   "central_channel": "Central channel %", "near_channel": "Near channel %"}),
                 hide_index=True, width="stretch")
