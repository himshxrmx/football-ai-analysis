"""Page: optional 2D pose estimation for one tracked player."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import video_processor as vp
from modules.player_detector import ModelLoadError
from ui import components, state


@st.cache_resource(show_spinner="Loading the pose model...")
def _pose_estimator():
    from modules.pose_estimator import PoseEstimator

    return PoseEstimator()


def render() -> None:
    components.page_header("Pose estimation (optional)", "Estimates 17 body keypoints for one tracked player in one frame.")
    st.info("Optional extension. The specification suggested MediaPipe, but current MediaPipe releases install a "
            "second OpenCV package that conflicts with the one used by YOLO. This page therefore uses the YOLO pose "
            "model from the same Ultralytics library. Angles are measured in the 2D image and depend on the camera "
            "angle, so they are indicative only. Players in wide broadcast shots are often too small for reliable "
            "keypoints.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    if not run.video_path.exists():
        st.warning("The original video is no longer at its saved location, so frames cannot be read.")
        return
    metrics = state.bundle(run)["metrics"]
    stable = state.stable_ids(metrics)
    if not stable:
        st.info("No stable tracks are available.")
        return
    tracks = run.tracks
    c1, c2 = st.columns(2)
    player = c1.selectbox("Track", stable, format_func=lambda i: f"ID {i}")
    rows = tracks[tracks["track_id"] == player].copy()
    rows["height"] = rows["y2"] - rows["y1"]
    rows = rows.sort_values("frame_idx").reset_index(drop=True)
    tallest = int(rows["height"].idxmax())
    position = c2.slider("Analysed frame of this track", 0, len(rows) - 1, tallest,
                         help="Starts at the frame where the player appears largest.")
    row = rows.iloc[position]
    st.caption(f"Video frame {int(row['frame_idx'])}, {row['time_s']:.2f} s, player box {row['height']:.0f} px tall.")
    if st.button("Estimate pose", type="primary"):
        try:
            estimator = _pose_estimator()
            frame = vp.read_frame(run.video_path, int(row["frame_idx"]))
            result = estimator.estimate(frame, (row["x1"], row["y1"], row["x2"], row["y2"]))
        except (ModelLoadError, vp.VideoError, ValueError) as exc:
            st.error(str(exc))
            return
        left, right = st.columns([1, 2])
        left.image(result.image_rgb, caption="Enlarged crop with detected keypoints", width="stretch")
        with right:
            st.write(result.message)
            if result.angles:
                st.dataframe(pd.DataFrame({"angle (degrees, 2D image)": result.angles}), width="stretch")
            if not result.keypoints.empty:
                with st.expander("Keypoint coordinates (frame pixels)"):
                    st.dataframe(result.keypoints, hide_index=True, width="stretch")
