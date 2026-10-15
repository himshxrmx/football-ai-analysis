"""Page: overall and individual heatmaps and movement paths."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from modules import heatmap_generator as hg
from ui import components, state


@st.cache_data(show_spinner=False, max_entries=4)
def _reference_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def render() -> None:
    components.page_header("Heatmaps and movement", "Where tracked people spent their time and how they moved.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    data = state.bundle(run)
    steps, metrics = data["steps"], data["metrics"]
    if steps.empty:
        st.info("No positions were recorded for this analysis.")
        return

    if run.is_calibrated:
        view = st.radio("View", ["Top-down pitch (calibrated, metres)", "Camera view (reference frame)"], horizontal=True)
        pitch_view = view.startswith("Top-down")
    else:
        pitch_view = False
        st.info("Showing the camera view. Calibrate the video on the **Pitch calibration** page to see a top-down "
                "pitch view in metres.")
    if pitch_view:
        # steps are already in pitch metres when the run is calibrated
        xy = steps
        cal = run.calibration
    else:
        if run.is_calibrated:
            st.caption("Camera view: positions are drawn in reference-frame pixels.")
        xy = steps.assign(sx=steps["ref_x"], sy=steps["ref_y"]) if run.is_calibrated else steps
        reference = _reference_rgb(str(run.reference_frame))

    stable = state.stable_ids(metrics)
    groups = sorted(g for g in metrics["colour_group"].unique() if g != "Unassigned")
    c1, c2 = st.columns([1, 2])
    group_filter = c1.multiselect("Colour groups", groups + ["Unassigned"], default=groups + ["Unassigned"],
                                  help="Groups are formed from shirt colours (experimental). Unassigned covers "
                                       "referees, staff and unclear colours.")
    eligible = metrics[metrics["stable"] & metrics["colour_group"].isin(group_filter)]["track_id"].astype(int).tolist()
    selected = c2.multiselect("Tracks included in the overall heatmap", eligible, default=eligible,
                              format_func=lambda i: f"ID {i}")

    st.header("Overall activity")
    subset = xy[xy["track_id"].isin(selected)]
    if subset.empty:
        st.info("No tracks selected.")
    elif pitch_view:
        components.plot(hg.pitch_heatmap(subset["sx"], subset["sy"], cal.pitch_length, cal.pitch_width), "overall_pitch")
    else:
        image, outside = hg.image_heatmap(reference, subset["sx"], subset["sy"])
        st.image(image, width="stretch")
        st.caption(f"Darker blue means more time spent. Background: the reference frame (the widest view in the clip). "
                   f"{outside * 100:.0f}% of positions fall outside this view and are not drawn.")
        st.download_button("Download heatmap image (PNG)", cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes(),
                           f"{run.run_id}_heatmap.png", "image/png")

    st.header("Individual heatmap")
    if not stable:
        st.info("No stable tracks are available.")
    else:
        player = st.selectbox("Track", stable, format_func=lambda i: f"ID {i}", key="heat_player")
        one = xy[xy["track_id"] == player]
        row = metrics.set_index("track_id").loc[player]
        st.caption(f"ID {player}: tracked for {row['visible_s']:.1f} s between {row['first_seen_s']:.1f} s and "
                   f"{row['last_seen_s']:.1f} s; colour group {row['colour_group']}.")
        if pitch_view:
            components.plot(hg.pitch_heatmap(one["sx"], one["sy"], cal.pitch_length, cal.pitch_width), "single_pitch")
        else:
            image, _ = hg.image_heatmap(reference, one["sx"], one["sy"], sigma_px=12)
            st.image(image, width="stretch")

    st.header("Movement paths")
    if stable:
        default = stable[:3]
        chosen = st.multiselect(f"Tracks (up to {hg.MAX_SERIES})", stable, default=default,
                                max_selections=hg.MAX_SERIES, format_func=lambda i: f"ID {i}", key="path_players")
        colours = components.series_colours(chosen, "paths")
        paths = {f"ID {i}": xy[xy["track_id"] == i].sort_values("time_s") for i in chosen}
        labelled = {f"ID {i}": colours[i] for i in chosen}
        if not chosen:
            st.info("Select at least one track.")
        elif pitch_view:
            components.plot(hg.pitch_paths(paths, labelled, cal.pitch_length, cal.pitch_width), "paths_pitch")
        else:
            components.plot(hg.image_paths(reference, paths, labelled), "paths_image")
        st.caption("Paths use smoothed positions. The dot with the label marks where the track ended.")
