"""Class mapping, model loading, detection, the pitch filter and tracking."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.player_detector import (
    Detections, ModelLoadError, PlayerDetector, compute_pitch_mask, load_yolo, on_pitch_mask, resolve_class_map,
)
from modules.player_tracker import PlayerTracker, TrailBuffer
from tests.conftest import requires_model


def test_coco_class_map():
    names = {0: "person", 1: "bicycle", 32: "sports ball"}
    cmap = resolve_class_map(names)
    assert cmap.person_ids == [0]
    assert cmap.ball_ids == [32]
    assert cmap.model_type == "coco"
    assert not cmap.can_separate_referees


def test_football_class_map():
    cmap = resolve_class_map({0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"})
    assert cmap.person_ids == [1, 2, 3]
    assert cmap.ball_ids == [0]
    assert cmap.model_type == "football"
    assert cmap.roles[3] == "referee"
    assert cmap.can_separate_referees


def test_model_without_people_is_rejected():
    with pytest.raises(ModelLoadError):
        resolve_class_map({0: "car", 1: "dog"})


def test_missing_custom_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_yolo(str(tmp_path / "football.pt"))


def test_corrupted_model_gives_clear_error(tmp_path):
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError):
        load_yolo(str(bad))


def test_detections_helpers():
    det = Detections(xyxy=np.array([[0, 0, 10, 20], [5, 5, 15, 25]], dtype=np.float32),
                     confidence=np.array([0.9, 0.2], dtype=np.float32),
                     class_id=np.array([0, 0]), tracker_id=np.array([-1, -1]))
    assert len(det) == 2
    np.testing.assert_allclose(det.foot_points, [[5, 20], [10, 25]])
    assert len(det.filter(det.confidence > 0.5)) == 1
    assert len(Detections()) == 0 and Detections().foot_points.shape == (0, 2)


def test_pitch_filter_keeps_people_on_grass():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)                      # grey stands
    frame[150:, :] = (40, 140, 40)               # green pitch in the lower part
    mask = compute_pitch_mask(frame)
    assert mask is not None
    det = Detections(xyxy=np.array([[100, 250, 120, 300], [100, 50, 120, 100]], dtype=np.float32),
                     confidence=np.array([0.9, 0.9], dtype=np.float32), class_id=np.zeros(2, dtype=np.int32),
                     tracker_id=np.full(2, -1, dtype=np.int32))
    keep = on_pitch_mask(det, mask)
    assert keep.tolist() == [True, False]


def test_pitch_filter_disabled_without_pitch():
    frame = np.full((300, 300, 3), 80, dtype=np.uint8)  # no grass at all
    assert compute_pitch_mask(frame) is None
    det = Detections(xyxy=np.array([[10, 10, 20, 40]], dtype=np.float32), confidence=np.array([0.5], dtype=np.float32),
                     class_id=np.zeros(1, dtype=np.int32), tracker_id=np.full(1, -1, dtype=np.int32))
    assert on_pitch_mask(det, None).tolist() == [True]


@requires_model
def test_detects_people_in_reference_image():
    import ultralytics

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    detector = PlayerDetector()
    people, _ = detector.detect(image, conf=0.3, imgsz=640)
    assert len(people) >= 3
    assert (people.confidence >= 0.3).all()


@requires_model
def test_empty_frame_returns_no_detections():
    detector = PlayerDetector()
    people, ball = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), conf=0.3, imgsz=640)
    assert len(people) == 0 and len(ball) == 0


def _box(x, y, conf=0.9):
    return Detections(xyxy=np.array([[x, y, x + 20, y + 50]], dtype=np.float32),
                      confidence=np.array([conf], dtype=np.float32), class_id=np.zeros(1, dtype=np.int32),
                      tracker_id=np.full(1, -1, dtype=np.int32))


def test_tracker_keeps_id_for_moving_box():
    tracker = PlayerTracker(effective_fps=25, confidence=0.3)
    ids = []
    for i in range(20):
        out = tracker.update(_box(100 + 3 * i, 100))
        if len(out):
            ids.append(int(out.tracker_id[0]))
    assert len(ids) >= 18
    assert len(set(ids)) == 1


def test_tracker_separates_two_players():
    tracker = PlayerTracker(effective_fps=25, confidence=0.3)
    seen = set()
    for i in range(10):
        a, b = _box(100 + 2 * i, 100), _box(400 - 2 * i, 300)
        both = Detections(xyxy=np.vstack([a.xyxy, b.xyxy]), confidence=np.array([0.9, 0.9], dtype=np.float32),
                          class_id=np.zeros(2, dtype=np.int32), tracker_id=np.full(2, -1, dtype=np.int32))
        out = tracker.update(both)
        seen.update(int(t) for t in out.tracker_id)
    assert len(seen) == 2


def test_tracker_handles_empty_frames():
    tracker = PlayerTracker(effective_fps=25, confidence=0.3)
    assert len(tracker.update(Detections())) == 0
    tracker.update(_box(100, 100))
    assert len(tracker.update(Detections())) == 0


def test_new_tracker_restarts_ids():
    first = PlayerTracker(effective_fps=25)
    for _ in range(3):
        out_a = first.update(_box(50, 50))
    second = PlayerTracker(effective_fps=25)
    for _ in range(3):
        out_b = second.update(_box(50, 50))
    assert int(out_b.tracker_id[0]) == int(out_a.tracker_id[0]) == 1


def test_trail_buffer_length_and_prune():
    trails = TrailBuffer(max_length=3)
    for i in range(5):
        trails.add(7, (i, i), i)
    assert trails.get(7) == [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    trails.prune(frame_number=20, max_age=5)
    assert trails.get(7) == []
