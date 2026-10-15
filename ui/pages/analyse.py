"""Page: upload or choose a video, validate it, check detection and run the analysis."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from modules import annotator
from modules import video_processor as vp
from modules.pipeline import AnalysisError, AnalysisSettings, plan_frames, run_analysis
from modules.player_detector import ModelLoadError, compute_pitch_mask, on_pitch_mask
from ui import components, state


def _duration_text(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


@st.cache_data(show_spinner="Checking the video...", max_entries=16)
def _validate(path: str, mtime: float) -> vp.ValidationResult:
    return vp.validate_video_file(path)


@st.cache_data(show_spinner="Preparing a browser preview...", max_entries=8)
def _preview(path: str, mtime: float) -> str | None:
    info = vp.get_video_info(path)
    if vp.is_browser_playable(info):
        return path
    target = config.UPLOADS_DIR / f"preview_{Path(path).stem}.mp4"
    result = vp.make_browser_preview(path, target)
    return str(result) if result else None


def _choose_video() -> str | None:
    samples = vp.list_sample_videos()
    source = st.radio("Video source", ["Upload a file", "Use a sample video"], horizontal=True)
    if source == "Upload a file":
        upload = st.file_uploader(
            "Football video",
            type=[ext.strip(".") for ext in config.ALLOWED_EXTENSIONS],
            help=f"MP4, AVI, MOV or MKV. Up to {config.MAX_FILE_SIZE_MB} MB and "
                 f"{config.MAX_DURATION_SECONDS // 60} minutes. Short clips (10 to 60 s) work best on a CPU.",
        )
        if upload is None:
            return None
        key = f"{upload.name}:{upload.size}"
        if st.session_state.get("upload_key") != key:
            if not vp.validate_extension(upload.name):
                st.error(f"'{upload.name}' is not a supported video type.")
                return None
            st.session_state["upload_path"] = str(vp.save_uploaded_bytes(upload.getvalue(), upload.name))
            st.session_state["upload_key"] = key
        return st.session_state["upload_path"]
    if not samples:
        st.info(f"No sample videos found. Copy an MP4 file into the folder {config.SAMPLES_DIR} and reload the page.")
        return None
    names = [p.name for p in samples]
    choice = st.selectbox("Sample video", names)
    return str(samples[names.index(choice)])


def _settings(info: vp.VideoInfo) -> AnalysisSettings:
    models = list(config.PRETRAINED_MODELS)
    if config.CUSTOM_MODEL_PATH.exists():
        models.append(str(config.CUSTOM_MODEL_PATH))
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox(
        "Detection model", models,
        format_func=lambda m: config.PRETRAINED_MODELS.get(m, f"Custom football model ({Path(m).name})"),
        help="COCO models detect every person and the ball. A custom football model can separate players, "
             "goalkeepers and referees.",
    )
    confidence = c1.slider(
        "Confidence threshold", 0.10, 0.90, config.DEFAULT_CONFIDENCE, 0.05,
        help="Detections below this score are not shown and cannot start a track. Lower values find more distant "
             "players but also more false detections.",
    )
    image_size = c2.select_slider(
        "Processing resolution (model input, px)", options=list(config.IMAGE_SIZES), value=config.DEFAULT_IMAGE_SIZE,
        help="Larger values find small, distant players more reliably but are slower. Detection time measured on the "
             "development laptop CPU (Intel i7-8750H) with YOLO26 nano: about 100 ms at 640 px, 165 ms at 960 px and "
             "275 ms at 1280 px per frame.",
    )
    stride = c2.slider(
        "Analyse every Nth frame", 1, 5, config.DEFAULT_FRAME_STRIDE,
        help="2 analyses every second frame. Higher values are faster, but players move further between analysed "
             "frames, which makes tracking harder.",
    )
    longest = max(1, int(math.floor(info.duration_s)))
    if longest > 1:
        max_seconds = c3.slider("Seconds to analyse, from the start", 1, longest,
                                min(config.DEFAULT_MAX_SECONDS, longest))
    else:
        max_seconds = 1
        c3.caption("The clip is about one second long and will be analysed completely.")
    with c3.expander("More options"):
        detect_ball = st.checkbox("Detect the ball", value=True,
                                  help="Uses the 'sports ball' class. The ball is small and often missed.")
        pitch_filter = st.checkbox("Ignore people off the grass", value=True,
                                   help="Removes detections whose feet are outside the green playing area "
                                        "(crowd, benches).")
        camera = st.checkbox("Compensate camera movement", value=True,
                             help="Estimates how the camera pans and zooms so positions stay comparable over time.")
        trail = st.slider("Trail length (analysed frames)", 5, 60, config.DEFAULT_TRAIL_LENGTH)
    return AnalysisSettings(model_name=model_name, confidence=confidence, image_size=int(image_size),
                            frame_stride=int(stride), max_seconds=float(max_seconds), detect_ball=detect_ball,
                            pitch_filter=pitch_filter, camera_compensation=camera, trail_length=int(trail))


def _detection_check(video_path: str, info: vp.VideoInfo, settings: AnalysisSettings) -> None:
    frame_no = st.slider("Frame to test", 0, max(0, info.frame_count - 1), min(info.frame_count - 1, int(info.fps * 5)))
    if st.button("Run detection on this frame"):
        try:
            detector = state.detector(settings.model_name)
            frame = vp.read_frame(video_path, frame_no)
            detector.predict_raw(frame, settings.confidence, settings.image_size, settings.detect_ball)  # warm-up
            start = time.perf_counter()
            raw = detector.predict_raw(frame, settings.confidence, settings.image_size, settings.detect_ball)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except (ModelLoadError, vp.VideoError) as exc:
            st.error(str(exc))
            return
        people, ball = detector.split(raw)
        removed = 0
        if settings.pitch_filter and len(people):
            keep = on_pitch_mask(people, compute_pitch_mask(frame))
            removed = int((~keep).sum())
            people = people.filter(keep)
        roles = [detector.role_of(c) for c in people.class_id]
        image = annotator.bgr_to_rgb(annotator.draw_detections(frame, people, ball, roles, None, True, None))
        table = pd.DataFrame({
            "class": [detector.class_map.names[int(c)] for c in people.class_id]
                     + [detector.class_map.names[int(c)] for c in ball.class_id],
            "confidence": list(people.confidence.round(3)) + list(ball.confidence.round(3)),
            "x1": list(people.xyxy[:, 0].round(0)) + list(ball.xyxy[:, 0].round(0)),
            "y1": list(people.xyxy[:, 1].round(0)) + list(ball.xyxy[:, 1].round(0)),
            "x2": list(people.xyxy[:, 2].round(0)) + list(ball.xyxy[:, 2].round(0)),
            "y2": list(people.xyxy[:, 3].round(0)) + list(ball.xyxy[:, 3].round(0)),
        })
        st.session_state["detection_check"] = {
            "key": (video_path, frame_no, settings.model_name, settings.confidence, settings.image_size,
                    settings.detect_ball, settings.pitch_filter),
            "image": vp.resize_to_width(image, 1280), "table": table, "people": len(people), "ball": len(ball),
            "removed": removed, "ms": elapsed_ms,
        }
        st.session_state.setdefault("measured_ms", {})[(settings.model_name, settings.image_size)] = elapsed_ms

    result = st.session_state.get("detection_check")
    if not result or result["key"][0] != video_path:
        st.caption("Tests the model and the confidence threshold on a single frame before the full analysis.")
        return
    if result["key"][1:] != (frame_no, settings.model_name, settings.confidence, settings.image_size,
                             settings.detect_ball, settings.pitch_filter):
        st.caption("The settings or the frame changed. Run the check again to update the result.")
    components.metric_row([
        ("People detected", str(result["people"]), "Detections at or above the confidence threshold."),
        ("Ball detections", str(result["ball"]), None),
        ("Removed off the grass", str(result["removed"]), "Removed by the pitch filter."),
        ("Detection time", f"{result['ms']:.0f} ms", "Model inference time for this frame on this computer."),
    ])
    st.image(result["image"], caption="Boxes show detections with confidence scores. Tracking IDs are assigned "
                                      "during the full analysis.", width="stretch")
    if result["table"].empty:
        st.info("No objects were detected in this frame at the chosen threshold.")
    else:
        with st.expander(f"Detections in this frame ({len(result['table'])})"):
            st.dataframe(result["table"], hide_index=True, width="stretch")


def _estimate(settings: AnalysisSettings, frames: int) -> str | None:
    measured = st.session_state.get("measured_ms", {}).get((settings.model_name, settings.image_size))
    if measured:
        # Tracking, camera compensation and video writing add work on top of detection;
        # the factor 1.7 was measured on the development laptop.
        return (f"Estimated time: about {_duration_text(frames * measured * 1.7 / 1000)} "
                f"(based on the detection check on this computer).")
    for run in state.runs():
        try:
            meta = json.loads((Path(run["run_dir"]) / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        s = meta.get("settings", {})
        if s.get("model_name") == settings.model_name and s.get("image_size") == settings.image_size:
            fps = meta["processing"].get("processing_fps")
            if fps:
                return (f"Estimated time: about {_duration_text(frames / fps)} "
                        f"(based on the previous analysis with the same model and resolution).")
    return None


def _finished_panel() -> None:
    run_dir = st.session_state.get("last_finished")
    if not run_dir or not (Path(run_dir) / "meta.json").exists():
        return
    run = state.current_run()
    if run is None or str(run.run_dir) != run_dir:
        return
    data = state.bundle(run)
    ov = data["overview"]
    proc = run.meta["processing"]
    st.success(f"Analysis finished in {_duration_text(proc['elapsed_s'])} "
               f"({proc['processing_fps']} frames per second). Results are saved in outputs/{run.run_id}.")
    components.metric_row([
        ("Frames analysed", str(proc["frames_processed"]), None),
        ("People tracked per frame", f"{ov['mean_tracked_per_frame']}", "Average over all analysed frames."),
        ("Stable tracks", str(ov["stable_tracks"]), f"Tracks followed for at least {config.MIN_TRACK_SECONDS:.0f} s."),
        ("Track IDs created", str(ov["unique_track_ids"]), "Usually higher than the number of people because IDs "
                                                           "change after occlusions."),
    ])
    st.video(str(run.annotated_video))
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download annotated video", run.annotated_video.read_bytes(), f"{run.run_id}_annotated.mp4",
                       "video/mp4")
    c2.download_button("Download tracking data (CSV)", (run.run_dir / "tracks.csv").read_bytes(),
                       f"{run.run_id}_tracks.csv", "text/csv")
    st.page_link(st.session_state["pages"]["overview"], label="Open the overview of this analysis")


def render() -> None:
    components.page_header(
        "Analyse a video",
        "Choose a football clip, check that it can be read, test detection on one frame and run the full analysis.",
    )
    _finished_panel()

    st.header("1. Choose a video")
    video_path = _choose_video()
    if not video_path:
        return
    validation = _validate(video_path, Path(video_path).stat().st_mtime)
    if not validation.ok:
        for error in validation.errors:
            st.error(error)
        return
    for warning in validation.warnings:
        st.warning(warning)
    info = validation.info

    st.header("2. Check the video")
    left, right = st.columns([3, 2])
    with left:
        preview = _preview(video_path, Path(video_path).stat().st_mtime)
        if preview:
            st.video(preview)
        else:
            st.info("This file cannot be previewed in the browser, but it can still be analysed.")
    with right:
        components.key_value_table([
            ("File", info.file_name),
            ("Resolution", f"{info.width} x {info.height} px"),
            ("Frame rate", f"{info.fps:.2f} frames per second"),
            ("Duration", f"{info.duration_s:.1f} s"),
            ("Frames", f"{info.frame_count:,}"),
            ("Codec", info.codec),
            ("File size", f"{info.size_mb:.1f} MB"),
        ])
        st.caption("The file passed all checks: supported type, size and duration limits, and frames can be decoded.")

    st.header("3. Settings")
    settings = _settings(info)

    st.header("4. Check detection on one frame")
    _detection_check(video_path, info, settings)

    st.header("5. Run the analysis")
    _, frames = plan_frames(info.fps, info.frame_count, settings.frame_stride, settings.max_seconds)
    st.write(f"{frames} frames will be analysed: the first {settings.max_seconds:.0f} s of the video, "
             f"every {settings.frame_stride} frame(s).")
    estimate = _estimate(settings, frames)
    st.caption(estimate or "Run the detection check above to get a time estimate for this computer.")
    if st.button("Run analysis", type="primary"):
        progress = st.progress(0.0, text="Loading the model")
        last_update = [0.0]

        def on_progress(p) -> None:
            now = time.perf_counter()
            if now - last_update[0] < 0.25 and p.processed < p.total:
                return
            last_update[0] = now
            progress.progress(p.fraction, text=f"Analysed {p.processed} of {p.total} frames, "
                                               f"{p.fps:.1f} frames per second, about {_duration_text(p.eta_s)} left")

        try:
            detector = state.detector(settings.model_name)
            run_dir = run_analysis(video_path, settings, on_progress, detector=detector)
        except (ModelLoadError, AnalysisError, vp.VideoError) as exc:
            progress.empty()
            st.error(str(exc))
            return
        except Exception as exc:  # keep the app usable and show what happened
            progress.empty()
            st.error(f"The analysis stopped because of an unexpected error: {exc}")
            return
        state.set_active_run(run_dir)
        state.clear_caches()
        st.session_state["last_finished"] = str(run_dir)
        st.rerun()
