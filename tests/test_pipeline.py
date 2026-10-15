"""End-to-end pipeline on a small synthetic video (needs the model weights)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from modules.pipeline import AnalysisError, AnalysisSettings, plan_frames, run_analysis
from tests.conftest import requires_model, write_video


def test_plan_frames():
    assert plan_frames(25, 750, 2, 30) == (750, 375)
    assert plan_frames(25, 750, 2, 10) == (250, 125)
    assert plan_frames(25, 100, 3, None) == (100, 34)


def test_invalid_video_raises(tmp_path, corrupted_video):
    with pytest.raises(AnalysisError):
        run_analysis(corrupted_video, AnalysisSettings(), outputs_dir=tmp_path)


@requires_model
def test_pipeline_without_players_completes(tmp_path):
    video = write_video(tmp_path / "empty_pitch.mp4", frames=20, fps=10, size=(320, 240), moving_box=False)
    settings = AnalysisSettings(image_size=640, frame_stride=1, max_seconds=None)
    progress = []
    run_dir = run_analysis(video, settings, progress.append, outputs_dir=tmp_path / "out")
    for name in ("meta.json", "tracks.csv", "frames.csv", "ball.csv", "annotated.mp4", "reference_frame.jpg"):
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["processing"]["frames_processed"] == 20
    frames = pd.read_csv(run_dir / "frames.csv")
    assert (frames["tracked_players"] == 0).all()
    assert progress[-1].processed == progress[-1].total == 20
