"""End-to-end video analysis pipeline.

validate -> read frames -> detect (YOLO) -> pitch filter -> track (ByteTrack)
-> camera motion compensation -> record positions -> annotate -> save outputs

Every analysis is stored in its own folder under outputs/:
    meta.json            video information, settings and processing summary
    tracks.csv           one row per tracked player per processed frame
    ball.csv             ball detections (no tracking IDs)
    frames.csv           per-frame counts and camera homographies
    annotated.mp4        video with boxes, IDs, confidence and trails
    reference_frame.jpg  raw frame used for heatmaps and calibration
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

import config
from modules import annotator
from modules.camera_motion import CameraMotionEstimator, frame_histogram, histogram_change, transform_points
from modules.player_detector import Detections, PlayerDetector, compute_pitch_mask, on_pitch_mask
from modules.player_tracker import PlayerTracker, TrailBuffer
from modules.team_classifier import grass_colour_lab, jersey_colour_lab
from modules.video_processor import VideoWriter, iter_frames, safe_file_name, validate_video_file

BALL_MIN_CONFIDENCE = 0.20
SCENE_CUT_THRESHOLD = 0.5


class AnalysisError(Exception):
    pass


@dataclass
class AnalysisSettings:
    model_name: str = config.DEFAULT_MODEL
    confidence: float = config.DEFAULT_CONFIDENCE
    image_size: int = config.DEFAULT_IMAGE_SIZE
    frame_stride: int = config.DEFAULT_FRAME_STRIDE
    max_seconds: float | None = config.DEFAULT_MAX_SECONDS
    detect_ball: bool = True
    pitch_filter: bool = True
    camera_compensation: bool = True
    trail_length: int = config.DEFAULT_TRAIL_LENGTH
    track_lost_seconds: float = config.TRACK_LOST_SECONDS


@dataclass
class Progress:
    processed: int
    total: int
    elapsed_s: float
    message: str

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.processed / self.total)

    @property
    def fps(self) -> float:
        return self.processed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float:
        return (self.total - self.processed) / self.fps if self.fps > 0 else float("nan")


def plan_frames(fps: float, frame_count: int, stride: int, max_seconds: float | None) -> tuple[int, int]:
    """Return (source frames to scan, frames that will be processed)."""
    scan = frame_count
    if max_seconds:
        scan = min(frame_count, int(math.ceil(max_seconds * fps)))
    return scan, int(math.ceil(scan / max(1, stride)))


def run_analysis(
    video_path: str | Path,
    settings: AnalysisSettings,
    progress_callback: Callable[[Progress], None] | None = None,
    outputs_dir: Path = config.OUTPUTS_DIR,
    detector: PlayerDetector | None = None,
) -> Path:
    """Analyse a video and return the folder that holds the results."""
    video_path = Path(video_path)
    validation = validate_video_file(video_path)
    if not validation.ok:
        raise AnalysisError(" ".join(validation.errors))
    info = validation.info

    detector = detector or PlayerDetector(settings.model_name)
    stride = max(1, int(settings.frame_stride))
    effective_fps = info.fps / stride
    scan_frames, total = plan_frames(info.fps, info.frame_count, stride, settings.max_seconds)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{Path(safe_file_name(info.file_name)).stem[:40]}"
    run_dir = Path(outputs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    tracker = PlayerTracker(effective_fps, settings.confidence, settings.track_lost_seconds)
    camera = CameraMotionEstimator(enabled=settings.camera_compensation)
    trails = TrailBuffer(settings.trail_length)
    detection_conf = min(config.TRACKER_LOW_CONFIDENCE, settings.confidence)
    corners = np.array([[0, 0], [info.width, 0], [info.width, info.height], [0, info.height]], dtype=np.float64)

    track_rows: list[dict] = []
    ball_rows: list[dict] = []
    frame_rows: list[dict] = []
    homographies: list[np.ndarray] = []
    best_view_area, reference_frame, reference_index = 0.0, None, 0
    prev_hist = None
    off_pitch_removed = 0
    start = time.perf_counter()
    processed = 0

    # The annotated video is written at up to OUTPUT_MAX_WIDTH pixels wide: smaller files,
    # faster encoding, and still sharp enough to read IDs.
    out_scale = min(1.0, config.OUTPUT_MAX_WIDTH / info.width)
    out_size = (int(round(info.width * out_scale)), int(round(info.height * out_scale)))
    writer = VideoWriter(run_dir / "annotated.mp4", effective_fps, out_size)
    try:
        for frame_idx, t, frame in iter_frames(video_path, stride=stride, max_frames=scan_frames):
            raw = detector.predict_raw(frame, detection_conf, settings.image_size, settings.detect_ball)
            people, ball = detector.split(raw)

            pitch_mask = compute_pitch_mask(frame) if settings.pitch_filter else None
            if settings.pitch_filter and len(people):
                keep = on_pitch_mask(people, pitch_mask)
                off_pitch_removed += int(np.sum(~keep & (people.confidence >= settings.confidence)))
                people = people.filter(keep)

            tracked = tracker.update(people, frame)
            h_to_first, camera_ok = camera.update(frame, people.xyxy)

            # The reference view is the widest (most zoomed-out) frame, so that as many
            # positions as possible fall inside it for heatmaps and calibration.
            view_area = cv2.contourArea(transform_points(corners, h_to_first).astype(np.float32))
            if view_area > best_view_area * 1.001:
                best_view_area, reference_frame, reference_index = view_area, frame.copy(), processed

            hist = frame_histogram(frame)
            change = histogram_change(prev_hist, hist)
            prev_hist = hist

            # Ball: keep the single most confident detection on the pitch.
            best_ball = Detections()
            if len(ball):
                ball_ok = ball.confidence >= BALL_MIN_CONFIDENCE
                if settings.pitch_filter:
                    ball_ok &= on_pitch_mask(ball, pitch_mask)
                if ball_ok.any():
                    idx = int(np.argmax(np.where(ball_ok, ball.confidence, -1)))
                    best_ball = ball.filter(np.arange(len(ball)) == idx)
                    bx1, by1, bx2, by2 = best_ball.xyxy[0]
                    centre = np.array([[(bx1 + bx2) / 2, (by1 + by2) / 2]])
                    first_xy = transform_points(centre, h_to_first)[0]
                    ball_rows.append({
                        "frame_idx": frame_idx, "time_s": round(t, 3),
                        "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                        "confidence": round(float(best_ball.confidence[0]), 4),
                        "img_x": centre[0, 0], "img_y": centre[0, 1],
                        "first_x": first_xy[0], "first_y": first_xy[1],
                    })

            feet = tracked.foot_points
            feet_first = transform_points(feet, h_to_first)
            roles = [detector.role_of(c) for c in tracked.class_id]
            grass_lab = grass_colour_lab(frame) if len(tracked) else None
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                colour = jersey_colour_lab(frame, tracked.xyxy[i], grass_lab)
                track_id = int(tracked.tracker_id[i])
                track_rows.append({
                    "frame_idx": frame_idx,
                    "time_s": round(t, 3),
                    "track_id": track_id,
                    "role": roles[i],
                    "confidence": round(float(tracked.confidence[i]), 4),
                    "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                    "foot_x": round(float(feet[i, 0]), 1), "foot_y": round(float(feet[i, 1]), 1),
                    "first_x": float(feet_first[i, 0]), "first_y": float(feet_first[i, 1]),
                    "jersey_l": None if colour is None else float(colour[0]),
                    "jersey_a": None if colour is None else float(colour[1]),
                    "jersey_b": None if colour is None else float(colour[2]),
                })
                trails.add(track_id, feet_first[i], processed)
            trails.prune(processed, max_age=int(settings.track_lost_seconds * effective_fps) + 1)

            # Trails are stored in first-frame coordinates and projected into the
            # current frame, so they stay attached to the ground while the camera pans.
            h_first_to_current = np.linalg.inv(h_to_first)
            visible_trails = {}
            for track_id in tracked.tracker_id:
                points = trails.get(int(track_id))
                if len(points) >= 2:
                    visible_trails[int(track_id)] = transform_points(np.asarray(points), h_first_to_current).tolist()

            n_confident = int(np.sum(people.confidence >= settings.confidence)) if len(people) else 0
            hud = f"Frame {frame_idx}   {t:6.1f} s   tracked players: {len(tracked)}"
            annotated = annotator.draw_detections(frame, tracked, best_ball, roles, visible_trails, True, hud)
            if out_scale < 1.0:
                annotated = cv2.resize(annotated, out_size, interpolation=cv2.INTER_AREA)
            writer.write(annotated)

            frame_rows.append({
                "frame_idx": frame_idx,
                "time_s": round(t, 3),
                "tracked_players": int(len(tracked)),
                "detections": n_confident,
                "ball_detected": int(len(best_ball) > 0),
                "camera_step_ok": int(camera_ok),
                "scene_change": round(change, 4),
            })
            homographies.append(h_to_first)
            processed += 1
            if progress_callback is not None:
                progress_callback(Progress(processed, total, time.perf_counter() - start,
                                           f"Frame {frame_idx} of {scan_frames}"))
    finally:
        writer.close()

    if processed == 0:
        raise AnalysisError("No frames could be read from the video.")
    cv2.imwrite(str(run_dir / "reference_frame.jpg"), reference_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Re-express every position in the reference frame's pixel coordinates.
    h_first_to_ref = np.linalg.inv(homographies[reference_index])
    tracks = pd.DataFrame(track_rows)
    if not tracks.empty:
        ref = transform_points(tracks[["first_x", "first_y"]].to_numpy(), h_first_to_ref)
        tracks["ref_x"], tracks["ref_y"] = np.round(ref[:, 0], 2), np.round(ref[:, 1], 2)
        tracks = tracks.drop(columns=["first_x", "first_y"])
    else:
        tracks = pd.DataFrame(columns=["frame_idx", "time_s", "track_id", "role", "confidence", "x1", "y1", "x2", "y2",
                                       "foot_x", "foot_y", "jersey_l", "jersey_a", "jersey_b", "ref_x", "ref_y"])
    balls = pd.DataFrame(ball_rows)
    if not balls.empty:
        ref = transform_points(balls[["first_x", "first_y"]].to_numpy(), h_first_to_ref)
        balls["ref_x"], balls["ref_y"] = np.round(ref[:, 0], 2), np.round(ref[:, 1], 2)
        balls = balls.drop(columns=["first_x", "first_y"])
    frames = pd.DataFrame(frame_rows)
    h_to_ref = np.stack([h_first_to_ref @ h for h in homographies])
    for r in range(3):
        for c in range(3):
            frames[f"h{r}{c}"] = h_to_ref[:, r, c]

    tracks.to_csv(run_dir / "tracks.csv", index=False)
    balls.to_csv(run_dir / "ball.csv", index=False)
    frames.to_csv(run_dir / "frames.csv", index=False)

    elapsed = time.perf_counter() - start
    scene_cuts = int((frames["scene_change"] > SCENE_CUT_THRESHOLD).sum())
    meta = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": info.to_dict(),
        "settings": asdict(settings),
        "model": {
            "name": detector.model_name,
            "type": detector.class_map.model_type,
            "classes_used": {str(k): detector.class_map.names[k] for k in detector.class_map.person_ids + detector.class_map.ball_ids},
            "can_separate_referees": detector.class_map.can_separate_referees,
        },
        "tracker": {k: v for k, v in tracker.settings.items() if isinstance(v, (int, float, str, bool))},
        "processing": {
            "frames_processed": processed,
            "source_frames_scanned": int(frames["frame_idx"].iloc[-1]) + 1,
            "effective_fps": round(effective_fps, 3),
            "duration_processed_s": round(float(frames["time_s"].iloc[-1]) + 1.0 / info.fps, 2),
            "elapsed_s": round(elapsed, 1),
            "processing_fps": round(processed / elapsed, 2) if elapsed > 0 else None,
            "video_writer": writer.backend,
            "camera_compensation": settings.camera_compensation,
            "camera_success_rate": round(camera.success_rate, 3),
            "scene_cuts": scene_cuts,
            "reference_processed_index": reference_index,
            "reference_frame_idx": int(frames["frame_idx"].iloc[reference_index]),
            "off_pitch_detections_removed": off_pitch_removed,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
