"""Page: camera-to-pitch calibration from manually entered landmarks."""
from __future__ import annotations

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from modules import heatmap_generator as hg
from modules.pitch_calibration import CalibrationError, compute_calibration, draw_projected_pitch, pitch_landmarks
from modules.run_store import remove_calibration, save_calibration
from ui import components, state


def _landmark_figure(landmarks: dict, length: float, width: float) -> go.Figure:
    fig = hg.pitch_figure(length, width, None, 380)
    names = list(landmarks)
    fig.add_trace(go.Scatter(
        x=[landmarks[n][0] for n in names], y=[landmarks[n][1] for n in names], mode="markers",
        marker=dict(size=8, color=hg.SERIES[1], line=dict(color=hg.SURFACE, width=2)), text=names,
        hovertemplate="%{text}<br>x %{x:.2f} m, y %{y:.2f} m<extra></extra>", showlegend=False,
    ))
    fig.add_annotation(x=length / 2, y=-2.2, text="far touchline", showarrow=False, font=dict(color=hg.MUTED, size=11))
    fig.add_annotation(x=length / 2, y=width + 2.2, text="near touchline (camera side)", showarrow=False,
                       font=dict(color=hg.MUTED, size=11))
    return fig


def render() -> None:
    components.page_header(
        "Pitch calibration",
        "Link at least four pitch landmarks in the reference frame to their real positions. The system then "
        "converts positions to metres and draws a top-down pitch view.",
    )
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    image_bgr = cv2.imread(str(run.reference_frame))
    if image_bgr is None:
        st.error("The reference frame of this analysis is missing.")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    proc = run.meta["processing"]
    st.info("How it works: the pitch is flat, so one 3 x 3 homography maps reference-frame pixels to pitch metres. "
            "It is estimated from the landmark pairs with OpenCV. Positions from other frames are first moved into "
            "the reference view by the camera motion compensation."
            + ("" if proc["camera_compensation"] else " Camera motion compensation was off for this analysis, so the "
               "calibration is only valid if the camera did not move."))

    st.header("1. Find landmarks in the reference frame")
    st.caption(f"Reference frame: video frame {proc['reference_frame_idx']} (the widest view in the clip). Hover over "
               "the image to read pixel coordinates.")
    components.plot(hg.image_figure(image_rgb, None, 560), "calib_image")

    st.header("2. Enter the landmark pairs")
    c1, c2 = st.columns(2)
    existing = run.calibration
    length = c1.number_input("Pitch length (m)", 90.0, 120.0,
                             float(existing.pitch_length if existing else config.PITCH_LENGTH_M), 0.5)
    width = c2.number_input("Pitch width (m)", 45.0, 90.0,
                            float(existing.pitch_width if existing else config.PITCH_WIDTH_M), 0.5)
    landmarks = pitch_landmarks(length, width)
    with st.expander("Where are the landmarks?"):
        components.plot(_landmark_figure(landmarks, length, width), "landmark_map")
        st.caption("Left and right are as seen from the main camera. Standard markings: penalty area 16.5 m deep and "
                   "40.32 m wide, goal area 5.5 m by 18.32 m, centre circle radius 9.15 m. If the real pitch size is "
                   "unknown, 105 x 68 m is the common standard.")

    if existing:
        start = pd.DataFrame({"landmark": existing.landmark_names,
                              "x_px": [p[0] for p in existing.image_points],
                              "y_px": [p[1] for p in existing.image_points]})
    else:
        start = pd.DataFrame({"landmark": [None] * 4, "x_px": [None] * 4, "y_px": [None] * 4})
    edited = st.data_editor(
        start, num_rows="dynamic", width="stretch", key=f"calib_editor_{run.run_id}",
        column_config={
            "landmark": st.column_config.SelectboxColumn("Landmark", options=list(landmarks), required=True, width="large"),
            "x_px": st.column_config.NumberColumn("x (pixels)", min_value=0, max_value=int(image_rgb.shape[1]), step=1),
            "y_px": st.column_config.NumberColumn("y (pixels)", min_value=0, max_value=int(image_rgb.shape[0]), step=1),
        },
    )
    rows = edited.dropna(subset=["landmark", "x_px", "y_px"])
    st.caption(f"{len(rows)} complete row(s). At least 4 are needed; 5 or more spread over the pitch give a more "
               "reliable result.")

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Compute and save calibration", type="primary"):
        try:
            calibration = compute_calibration(
                rows[["x_px", "y_px"]].to_numpy(dtype=float).tolist(),
                [landmarks[n] for n in rows["landmark"]],
                rows["landmark"].tolist(), length, width,
            )
        except CalibrationError as exc:
            st.error(str(exc))
        else:
            save_calibration(run.run_dir, calibration)
            state.clear_caches()
            st.session_state["calibration_saved"] = run.run_id
            st.rerun()
    if existing and c2.button("Remove calibration"):
        remove_calibration(run.run_dir)
        state.clear_caches()
        st.rerun()

    if existing:
        st.header("3. Check the result")
        if st.session_state.get("calibration_saved") == run.run_id:
            st.success("Calibration saved. All pages now use metres and the top-down pitch view.")
        error = existing.reprojection_error_m
        quality = "good" if error < 0.5 else "acceptable" if error < 1.5 else "poor, check the points"
        components.metric_row([
            ("Landmarks used", str(len(existing.landmark_names)), None),
            ("Mean reprojection error", f"{error:.2f} m", "Distance between each landmark's known position and "
             "where the homography maps its pixel. With exactly four points this is always close to zero, so "
             "use the overlay below to judge the result."),
            ("Assessment", quality, None),
        ])
        overlay = draw_projected_pitch(image_bgr, existing)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width="stretch",
                 caption="Yellow: the pitch markings projected back into the image with the calibration. Red circles: "
                         "the entered points. The yellow lines should lie on the real lines.")
