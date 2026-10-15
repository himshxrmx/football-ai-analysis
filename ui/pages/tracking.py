"""Page: annotated video, frame-by-frame detections and track timeline."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules import annotator
from modules import heatmap_generator as hg
from modules import video_processor as vp
from modules.player_detector import Detections
from ui import components, state


@st.cache_data(show_spinner=False, max_entries=32)
def _frame_image(video_path: str, frame_idx: int, rows: list[dict], ball: list[dict]) -> np.ndarray:
    frame = vp.read_frame(video_path, frame_idx)
    players = Detections(
        xyxy=np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([r["confidence"] for r in rows], dtype=np.float32),
        class_id=np.zeros(len(rows), dtype=np.int32),
        tracker_id=np.array([r["track_id"] for r in rows], dtype=np.int32),
    )
    ball_det = Detections(
        xyxy=np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in ball], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([b["confidence"] for b in ball], dtype=np.float32),
        class_id=np.zeros(len(ball), dtype=np.int32),
        tracker_id=np.full(len(ball), -1, dtype=np.int32),
    )
    roles = [r["role"] for r in rows]
    image = annotator.draw_detections(frame, players, ball_det, roles, None, True, None)
    return vp.resize_to_width(annotator.bgr_to_rgb(image), 1280)


def render() -> None:
    components.page_header("Player tracking", "The annotated video, the detections in any analysed frame and how long each ID was tracked.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    st.info("Tracking IDs are numbers assigned by the tracker. They do not identify real players or shirt numbers, "
            "and a player can receive a new ID after being hidden.")

    st.header("Annotated video")
    if run.annotated_video.exists():
        st.video(str(run.annotated_video))
        st.download_button("Download annotated video", run.annotated_video.read_bytes(),
                           f"{run.run_id}_annotated.mp4", "video/mp4")
        st.caption("Boxes and labels show the tracking ID and detection confidence. Lines behind players are their "
                   "recent movement trails. A white triangle marks a ball detection.")
    else:
        st.warning("The annotated video file is missing from the results folder.")

    st.header("Frame by frame")
    frames = run.frames
    tracks = run.tracks
    if frames.empty:
        st.info("No frames were analysed.")
        return
    frame_ids = frames["frame_idx"].astype(int).tolist()
    position = st.slider("Analysed frame", 0, len(frame_ids) - 1, 0,
                         format="%d", help="Moves through the analysed frames in order.")
    frame_idx = frame_ids[position]
    time_s = float(frames["time_s"].iloc[position])
    rows = tracks[tracks["frame_idx"] == frame_idx] if not tracks.empty else tracks
    ball = run.ball[run.ball["frame_idx"] == frame_idx] if not run.ball.empty else run.ball
    st.caption(f"Video frame {frame_idx}, time {time_s:.2f} s, {len(rows)} tracked people, "
               f"{len(ball)} ball detection(s).")
    if run.video_path.exists():
        image = _frame_image(str(run.video_path), frame_idx,
                             rows[["x1", "y1", "x2", "y2", "confidence", "track_id", "role"]].to_dict("records"),
                             ball[["x1", "y1", "x2", "y2", "confidence"]].to_dict("records") if len(ball) else [])
        st.image(image, width="stretch")
    else:
        st.warning("The original video is no longer at its saved location, so the frame cannot be shown. "
                   "The detections are listed below.")
    if rows.empty:
        st.info("No people were tracked in this frame.")
    else:
        table = rows[["track_id", "role", "confidence", "x1", "y1", "x2", "y2", "foot_x", "foot_y"]].rename(columns={
            "track_id": "ID", "foot_x": "feet x", "foot_y": "feet y"})
        st.dataframe(table.sort_values("ID"), hide_index=True, width="stretch")

    st.header("Track timeline")
    if tracks.empty:
        st.info("No tracks were created.")
        return
    metrics = state.bundle(run)["metrics"]
    show_all = st.toggle("Include short tracks", value=False,
                         help="Short tracks are often fragments caused by occlusion or false detections.")
    ids = metrics["track_id"] if show_all else metrics.loc[metrics["stable"], "track_id"]
    subset = tracks[tracks["track_id"].isin(ids)]
    fig = go.Figure(go.Scattergl(
        x=subset["time_s"], y=subset["track_id"].astype(str), mode="markers",
        marker=dict(size=4, color=hg.SERIES[0]),
        hovertemplate="ID %{y}<br>%{x:.1f} s<extra></extra>",
    ))
    hg.base_layout(fig, None, max(320, min(1400, 16 * len(ids) + 80)))
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="track ID", type="category", categoryorder="array",
                     categoryarray=[str(i) for i in sorted(ids)])
    components.plot(fig, "timeline")
    st.caption(f"Each dot is an analysed frame in which the ID was tracked. {len(ids)} IDs shown.")
