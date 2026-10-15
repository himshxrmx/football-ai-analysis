"""Camera motion, calibration, movement statistics, colour groups, coaching rules and reports."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from modules import coaching_insights as ci
from modules import movement_analyzer as ma
from modules.camera_motion import CameraMotionEstimator, frame_histogram, histogram_change, transform_points
from modules.pitch_calibration import (
    Calibration, CalibrationError, compute_calibration, draw_projected_pitch, image_to_pitch, pitch_landmarks,
)
from modules.report_generator import _latin1, build_pdf_report
from modules.team_classifier import assign_colour_groups, jersey_colour_lab
from tests.conftest import make_run, straight_track


# ---------------------------------------------------------------- camera motion
def _textured(seed=1, size=(480, 640)):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (size[0] // 8, size[1] // 8), dtype=np.uint8)
    noise = cv2.resize(noise, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    img = cv2.merge([noise // 3, np.clip(noise.astype(int) // 2 + 90, 0, 255).astype(np.uint8), noise // 3])
    return img  # greenish texture, so it counts as grass


def test_camera_translation_is_recovered():
    base = _textured(size=(560, 760))
    first = base[40:520, 40:680]
    second = base[40:520, 52:692]  # camera panned right: content moves 12 px to the left
    est = CameraMotionEstimator(min_points=15)
    est.update(first)
    h, ok = est.update(second)
    assert ok
    mapped = transform_points(np.array([[100.0, 200.0]]), h)[0]
    np.testing.assert_allclose(mapped, [112.0, 200.0], atol=1.5)


def test_camera_disabled_returns_identity():
    est = CameraMotionEstimator(enabled=False)
    h, ok = est.update(_textured())
    assert ok and np.allclose(h, np.eye(3))


def test_scene_cut_measure():
    a = _textured(1)
    b = np.full_like(a, (200, 30, 30))
    assert histogram_change(frame_histogram(a), frame_histogram(a)) < 0.01
    assert histogram_change(frame_histogram(a), frame_histogram(b)) > 0.5
    assert histogram_change(None, frame_histogram(a)) == 0.0


# ---------------------------------------------------------------- calibration
def test_calibration_recovers_known_homography():
    marks = pitch_landmarks()
    names = ["Corner, far left", "Corner, far right", "Corner, near right", "Corner, near left", "Centre spot"]
    true_h = np.array([[8.0, 3.0, 100.0], [0.2, 9.0, 50.0], [0.0001, 0.004, 1.0]])
    pitch = np.array([marks[n] for n in names])
    image = cv2.perspectiveTransform(pitch.reshape(-1, 1, 2), true_h).reshape(-1, 2)
    cal = compute_calibration(image.tolist(), pitch.tolist(), names)
    assert cal.reprojection_error_m < 0.01
    np.testing.assert_allclose(image_to_pitch(image, cal), pitch, atol=0.01)
    overlay = draw_projected_pitch(np.zeros((1200, 1600, 3), dtype=np.uint8), cal)
    assert overlay.any()


def test_calibration_rejects_bad_input(tmp_path):
    marks = pitch_landmarks()
    with pytest.raises(CalibrationError):
        compute_calibration([[0, 0], [1, 1], [2, 2]], [marks["Centre spot"]] * 3, ["a", "b", "c"])
    line = ["Corner, far left", "Halfway line, far touchline", "Corner, far right", "Left penalty area, far corner on goal line"]
    line_pts = [marks[n] for n in line[:3]] + [(10.0, 0.0)]
    with pytest.raises(CalibrationError, match="one line"):
        compute_calibration([[0, 0], [100, 5], [200, 0], [50, 300]], line_pts, line)
    with pytest.raises(CalibrationError, match="once"):
        compute_calibration([[0, 0]] * 4, [marks["Centre spot"]] * 4, ["Centre spot"] * 4)


def test_calibration_save_and_load(tmp_path):
    cal = Calibration(["a"], [[1, 2]], [[3, 4]], 105, 68, np.eye(3).tolist(), 0.1, "now")
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal


# ---------------------------------------------------------------- movement
def test_distance_and_speed_in_metres(tmp_path):
    # 4 m/s to the right for 100 frames at 25 fps = 3.96 s -> 15.84 m
    tracks = straight_track(1, (20, 30), (4, 0), 100)
    run = make_run(tmp_path, tracks, calibrated=True)
    metrics = ma.player_metrics(run)
    row = metrics.iloc[0]
    assert row["unit"] == "m"
    assert abs(row["distance"] - 15.84) < 0.3
    assert abs(row["avg_speed"] - 4.0) < 0.1
    assert row["sprints"] == 0 and row["high_speed_distance"] == 0
    assert bool(row["stable"])


def test_sprint_and_high_speed_detection(tmp_path):
    tracks = straight_track(1, (5, 30), (8, 0), 60)   # 8 m/s for 2.4 s
    run = make_run(tmp_path, tracks, calibrated=True)
    row = ma.player_metrics(run).iloc[0]
    assert row["sprints"] == 1
    assert row["high_speed_distance"] > 15


def test_impossible_speed_is_rejected(tmp_path):
    a = straight_track(1, (10, 30), (2, 0), 50)
    b = straight_track(1, (60, 30), (2, 0), 50, start_frame=50)  # teleports 48 m in one frame
    run = make_run(tmp_path, pd.concat([a, b], ignore_index=True), calibrated=True)
    row = ma.player_metrics(run).iloc[0]
    assert row["rejected_steps"] >= 1
    assert row["distance"] < 10


def test_gaps_are_not_counted_as_movement(tmp_path):
    a = straight_track(1, (10, 30), (0, 0), 25)
    b = straight_track(1, (40, 30), (0, 0), 25, start_frame=200)  # reappears 30 m away after 7 s
    run = make_run(tmp_path, pd.concat([a, b], ignore_index=True), calibrated=True, frames_processed=50)
    row = ma.player_metrics(run).iloc[0]
    assert row["distance"] < 1.0


def test_uncalibrated_values_are_pixels(tmp_path):
    run = make_run(tmp_path, straight_track(1, (100, 500), (50, 0), 50))
    row = ma.player_metrics(run).iloc[0]
    assert row["unit"] == "px"
    assert np.isnan(row["sprints"]) and np.isnan(row["high_speed_distance"])


def test_empty_tracks_do_not_crash(tmp_path):
    empty = straight_track(1, (0, 0), (0, 0), 1).iloc[0:0]
    run = make_run(tmp_path, empty, frames_processed=0)
    metrics = ma.player_metrics(run)
    assert metrics.empty
    ov = ma.overview(run, metrics)
    assert ov["unique_track_ids"] == 0 and ov["stable_tracks"] == 0
    assert ma.zone_shares(run).empty
    assert ma.group_spread(ma.movement_table(run), metrics).empty
    assert ci.team_insights(metrics, pd.DataFrame(), ov, "image")[0].rule_id == "Q0"


def test_zone_shares(tmp_path):
    run = make_run(tmp_path, straight_track(1, (10, 10), (0, 0), 50), calibrated=True)
    zones = ma.zone_shares(run).iloc[0]
    assert zones["left_third"] == 100.0 and zones["far_channel"] == 100.0


# ---------------------------------------------------------------- colour groups
def test_colour_groups_split_two_kits_and_leave_outlier():
    rows = []
    kits = {1: (230, 125, 125), 2: (232, 126, 124), 3: (228, 124, 126), 4: (225, 88, 168), 5: (230, 86, 170),
            6: (227, 90, 166), 7: (45, 128, 128)}
    for tid, lab in kits.items():
        for _ in range(6):
            rows.append({"track_id": tid, "jersey_l": lab[0], "jersey_a": lab[1], "jersey_b": lab[2]})
    groups = assign_colour_groups(pd.DataFrame(rows)).set_index("track_id")["colour_group"]
    assert groups[1] == groups[2] == groups[3]
    assert groups[4] == groups[5] == groups[6]
    assert groups[1] != groups[4]
    assert groups[7] == "Unassigned"


def test_jersey_colour_ignores_tiny_boxes_and_grass():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)
    assert jersey_colour_lab(frame, (10, 10, 13, 15)) is None
    frame[20:60, 40:60] = (0, 0, 255)  # red shirt
    grass = cv2.cvtColor(np.full((1, 1, 3), (40, 140, 40), dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab = jersey_colour_lab(frame, (35, 10, 65, 90), grass)
    red = cv2.cvtColor(np.full((1, 1, 3), (0, 0, 255), dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    assert np.allclose(lab, red, atol=3)


# ---------------------------------------------------------------- coaching
def _group_run(tmp_path, slow_speed: float):
    parts = [straight_track(i, (10 + 5 * i, 20), (3.0, 0), 200) for i in range(1, 6)]
    parts.append(straight_track(9, (60, 40), (slow_speed, 0), 200))
    return make_run(tmp_path, pd.concat(parts, ignore_index=True), calibrated=True)


def test_slow_player_gets_suggestion_with_evidence(tmp_path):
    run = _group_run(tmp_path, slow_speed=1.0)
    metrics, zones = ma.player_metrics(run), ma.zone_shares(run)
    insights = ci.player_insights(9, metrics, zones, "pitch", 8.0)
    kinds = {i.rule_id: i for i in insights}
    assert "S1" in kinds
    s1 = kinds["S1"]
    assert s1.kind == "suggestion" and s1.drill and s1.caveat
    assert s1.evidence["ratio"] < 0.75
    assert any(i.kind == "observation" for i in insights)


def test_no_suggestions_without_calibration(tmp_path):
    parts = [straight_track(i, (100 + 50 * i, 500), (60, 0), 200) for i in range(1, 6)]
    run = make_run(tmp_path, pd.concat(parts, ignore_index=True), calibrated=False)
    insights = ci.player_insights(1, ma.player_metrics(run), ma.zone_shares(run), "image", 8.0)
    assert not [i for i in insights if i.kind == "suggestion"]
    assert any(i.rule_id == "Q5" for i in insights)


def test_short_track_gets_data_quality_note(tmp_path):
    parts = [straight_track(i, (10 + 5 * i, 20), (3.0, 0), 200) for i in range(1, 6)]
    parts.append(straight_track(9, (60, 40), (0.5, 0), 60))  # only 2.4 s of an 8 s clip
    run = make_run(tmp_path, pd.concat(parts, ignore_index=True), calibrated=True, frames_processed=200)
    insights = ci.player_insights(9, ma.player_metrics(run), ma.zone_shares(run), "pitch", 8.0)
    assert any(i.rule_id == "Q1" for i in insights)
    assert not [i for i in insights if i.kind == "suggestion"]


def test_zone_suggestion_needs_a_minute_of_tracking(tmp_path):
    # Six players standing still in the left third for 80 s (2000 frames at 25 fps).
    long_parts = [straight_track(i, (10 + i, 10 + 8 * i), (0.0, 0), 2000) for i in range(1, 7)]
    run = make_run(tmp_path, pd.concat(long_parts, ignore_index=True), calibrated=True)
    insights = ci.player_insights(1, ma.player_metrics(run), ma.zone_shares(run), "pitch", 80.0)
    assert any(i.rule_id == "S3" for i in insights)

    short_parts = [straight_track(i, (10 + i, 10 + 8 * i), (0.0, 0), 200) for i in range(1, 7)]
    run = make_run(tmp_path / "short", pd.concat(short_parts, ignore_index=True), calibrated=True)
    insights = ci.player_insights(1, ma.player_metrics(run), ma.zone_shares(run), "pitch", 8.0)
    assert not any(i.rule_id == "S3" for i in insights)


def test_unassigned_colour_gets_note(tmp_path):
    run = _group_run(tmp_path, slow_speed=1.0)
    metrics = ma.player_metrics(run)
    metrics.loc[metrics["track_id"] == 9, "colour_group"] = "Unassigned"
    insights = ci.player_insights(9, metrics, ma.zone_shares(run), "pitch", 8.0)
    assert any(i.rule_id == "Q6" for i in insights)


def test_insights_never_mention_unmeasured_events(tmp_path):
    run = _group_run(tmp_path, slow_speed=0.5)
    metrics, zones = ma.player_metrics(run), ma.zone_shares(run)
    text = " ".join(i.detail.lower() for i in ci.player_insights(9, metrics, zones, "pitch", 8.0))
    for word in ("pass", "shot", "goal", "possession", "fatigue", "injur", "tired"):
        assert word not in text


# ---------------------------------------------------------------- report
def test_pdf_report_is_created(tmp_path):
    run = _group_run(tmp_path, slow_speed=1.0)
    steps = ma.movement_table(run)
    metrics = ma.player_metrics(run, steps)
    ov = ma.overview(run, metrics)
    team = ci.team_insights(metrics, ma.group_spread(steps, metrics), ov, "pitch")
    players = {9: ci.player_insights(9, metrics, ma.zone_shares(run), "pitch", 8.0)}
    out = build_pdf_report(run, metrics, ov, steps, team, players, tmp_path / "report.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 10_000


def test_pdf_report_with_no_tracks(tmp_path):
    empty = straight_track(1, (0, 0), (0, 0), 1).iloc[0:0]
    run = make_run(tmp_path, empty, frames_processed=0)
    metrics = ma.player_metrics(run)
    ov = ma.overview(run, metrics)
    out = build_pdf_report(run, metrics, ov, ma.movement_table(run), ci.team_insights(metrics, pd.DataFrame(), ov, "image"),
                           {}, tmp_path / "empty.pdf")
    assert out.read_bytes().startswith(b"%PDF")


def test_latin1_sanitiser():
    assert _latin1("a — b ≥ c ’") == "a - b >= c '"
    assert _latin1("中") == "?"
