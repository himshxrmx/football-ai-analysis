"""Page: summary of the active analysis."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import config
from modules import heatmap_generator as hg
from ui import components, state


def render() -> None:
    components.page_header("Overview", "Video information, processing status and tracking statistics for the active analysis.")
    run = state.current_run()
    if run is None:
        components.no_run_message()
        return
    components.run_caption(run)
    data = state.bundle(run)
    ov = data["overview"]
    meta = run.meta
    video, proc, settings = meta["video"], meta["processing"], meta["settings"]

    st.header("Tracking statistics")
    components.metric_row([
        ("Track IDs created", str(ov["unique_track_ids"]),
         "Every new track gets a new ID. A person hidden for longer than the track buffer gets a new ID, so this "
         "number is usually higher than the number of people."),
        ("Stable tracks", str(ov["stable_tracks"]),
         f"Tracks followed for at least {config.MIN_TRACK_SECONDS:.0f} s; these are used in player statistics."),
        ("People tracked per frame", str(ov["mean_tracked_per_frame"]), "Average over all analysed frames."),
        ("Most people in one frame", str(ov["max_tracked_in_frame"]), None),
    ])
    components.metric_row([
        ("Empty frames", f"{ov['frames_without_players']} ({ov['frames_without_players_pct']}%)",
         "Analysed frames in which nobody was tracked."),
        ("Mean confidence", components.fmt(ov["mean_confidence"], 2), "Average model score of tracked boxes."),
        ("Frames with ball found", f"{ov['ball_detection_pct']}%",
         "Share of analysed frames with a ball detection. The ball is small and often hidden, so it is missed "
         "in many frames." if settings["detect_ball"] else "Ball detection was switched off."),
        ("Camera motion estimated", f"{ov['camera_success_pct']}%" if proc["camera_compensation"] else "off",
         "Share of frame-to-frame camera motion estimates that passed the quality checks."),
    ])

    notes = []
    if meta["model"]["type"] == "coco":
        notes.append("The COCO model detects every person. Referees, goalkeepers, coaches and ball crew are counted "
                     "as people; the pitch filter removed "
                     f"{proc['off_pitch_detections_removed']} detections standing off the grass.")
    if ov["scene_cuts"]:
        notes.append(f"{ov['scene_cuts']} scene cut(s) were detected. Positions after a cut may not line up with "
                     "positions before it.")
    if not run.is_calibrated:
        notes.append("The video is not calibrated to the pitch, so movement is measured in pixels. Calibrate it on "
                     "the Pitch calibration page to get estimates in metres and a top-down pitch view.")
    for note in notes:
        st.info(note)

    st.header("People tracked over time")
    frames = run.frames
    if frames.empty:
        st.info("No frames were analysed.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames["time_s"], y=frames["tracked_players"], mode="lines", name="Tracked",
                                 line=dict(color=hg.SERIES[0], width=2),
                                 hovertemplate="%{x:.1f} s: %{y} tracked<extra></extra>"))
        fig.add_trace(go.Scatter(x=frames["time_s"], y=frames["detections"], mode="lines",
                                 name=f"Detected (confidence at least {settings['confidence']:.2f})",
                                 line=dict(color=hg.SERIES[1], width=2),
                                 hovertemplate="%{x:.1f} s: %{y} detected<extra></extra>"))
        hg.base_layout(fig, None, 320)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(title="time (s)")
        fig.update_yaxes(title="people", rangemode="tozero")
        components.plot(fig, "overview_timeline")
        st.caption("Tracked people can be fewer than detections because a new detection must be confirmed before a "
                   "track starts, and more when a track is kept alive by a low-confidence detection.")

    st.header("Video and processing")
    left, right = st.columns(2)
    with left:
        components.key_value_table([
            ("File", video["file_name"]),
            ("Resolution", f"{video['width']} x {video['height']} px"),
            ("Frame rate", f"{video['fps']:.2f} frames per second"),
            ("Duration", f"{video['duration_s']:.1f} s ({video['frame_count']:,} frames)"),
            ("Codec / size", f"{video['codec']}, {video['size_mb']:.1f} MB"),
        ])
    with right:
        components.key_value_table([
            ("Status", "Completed"),
            ("Analysed", f"{proc['duration_processed_s']:.1f} s, every {settings['frame_stride']} frame(s), "
                         f"{proc['frames_processed']} frames"),
            ("Model", f"{meta['model']['name']}, input {settings['image_size']} px, "
                      f"confidence {settings['confidence']:.2f}"),
            ("Classes used", ", ".join(meta["model"]["classes_used"].values())),
            ("Processing time", f"{proc['elapsed_s']:.0f} s ({proc['processing_fps']} frames per second)"),
            ("Pitch calibration", "yes" if run.is_calibrated else "no"),
        ])
