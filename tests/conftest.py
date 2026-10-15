"""Shared fixtures. Synthetic videos and runs keep the tests fast and independent of real footage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.pitch_calibration import Calibration  # noqa: E402
from modules.run_store import load_run  # noqa: E402

MODEL_AVAILABLE = (config.MODELS_DIR / config.DEFAULT_MODEL).exists()
requires_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="model weights not downloaded yet")


def write_video(path: Path, frames: int = 30, fps: float = 10.0, size=(320, 240), moving_box: bool = True) -> Path:
    """Green 'pitch' with an optional white box moving to the right."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    w, h = size
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 140, 40)
        if moving_box:
            x = 20 + 4 * i
            cv2.rectangle(frame, (x, 100), (x + 20, 150), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    return write_video(tmp_path / "clip.mp4")


@pytest.fixture
def corrupted_video(tmp_path) -> Path:
    path = tmp_path / "broken.mp4"
    path.write_bytes(np.random.default_rng(0).integers(0, 255, 50_000, dtype=np.uint8).tobytes())
    return path


def make_run(tmp_path: Path, tracks: pd.DataFrame, calibrated: bool = False, fps: float = 25.0, stride: int = 1,
             frames_processed: int | None = None) -> object:
    """Write a minimal analysis folder and load it with the real loader."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    effective = fps / stride
    if frames_processed is None:
        frames_processed = int(tracks["frame_idx"].nunique()) if not tracks.empty else 0
    frame_ids = sorted(tracks["frame_idx"].unique()) if not tracks.empty else []
    frames = pd.DataFrame({
        "frame_idx": frame_ids,
        "time_s": [f / fps for f in frame_ids],
        "tracked_players": [int((tracks["frame_idx"] == f).sum()) for f in frame_ids],
        "detections": [int((tracks["frame_idx"] == f).sum()) for f in frame_ids],
        "ball_detected": [0] * len(frame_ids),
        "camera_step_ok": [1] * len(frame_ids),
        "scene_change": [0.0] * len(frame_ids),
    })
    meta = {
        "run_id": "test_run", "created": "2026-10-15 12:00:00",
        "video": {"path": str(tmp_path / "missing.mp4"), "file_name": "test.mp4", "width": 1920, "height": 1080,
                  "fps": fps, "frame_count": 1000, "duration_s": 40.0, "codec": "h264", "size_mb": 1.0},
        "settings": {"model_name": "yolo26n.pt", "confidence": 0.3, "image_size": 960, "frame_stride": stride,
                     "max_seconds": 30, "detect_ball": True, "pitch_filter": True, "camera_compensation": True,
                     "trail_length": 25, "track_lost_seconds": 2.0},
        "model": {"name": "yolo26n.pt", "type": "coco", "classes_used": {"0": "person"}, "can_separate_referees": False},
        "processing": {"frames_processed": frames_processed, "effective_fps": effective,
                       "duration_processed_s": frames_processed / effective, "elapsed_s": 1.0, "processing_fps": 1.0,
                       "camera_compensation": True, "camera_success_rate": 1.0, "scene_cuts": 0,
                       "reference_frame_idx": 0, "off_pitch_detections_removed": 0, "video_writer": "test"},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    tracks.to_csv(run_dir / "tracks.csv", index=False)
    frames.to_csv(run_dir / "frames.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "ball.csv", index=False)
    cv2.imwrite(str(run_dir / "reference_frame.jpg"), np.full((1080, 1920, 3), 90, dtype=np.uint8))
    if calibrated:
        # Identity homography: reference pixels are treated as metres, which makes expected values easy to compute.
        Calibration(["a", "b", "c", "d"], [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [1, 0], [1, 1], [0, 1]],
                    105.0, 68.0, np.eye(3).tolist(), 0.0, "2026-10-15").save(run_dir / "calibration.json")
    return load_run(run_dir)


def straight_track(track_id: int, start_xy, velocity_xy, frames: int, fps: float = 25.0, stride: int = 1,
                   start_frame: int = 0) -> pd.DataFrame:
    """A track moving at constant velocity (units per second) in reference coordinates."""
    rows = []
    for k in range(frames):
        f = start_frame + k * stride
        t = f / fps
        x = start_xy[0] + velocity_xy[0] * (t - start_frame / fps)
        y = start_xy[1] + velocity_xy[1] * (t - start_frame / fps)
        rows.append({"frame_idx": f, "time_s": t, "track_id": track_id, "role": "person", "confidence": 0.8,
                     "x1": x - 5, "y1": y - 30, "x2": x + 5, "y2": y, "foot_x": x, "foot_y": y,
                     "jersey_l": 200.0, "jersey_a": 128.0, "jersey_b": 128.0, "ref_x": x, "ref_y": y})
    return pd.DataFrame(rows)
