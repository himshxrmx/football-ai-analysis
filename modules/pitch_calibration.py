"""Camera-to-pitch calibration with a planar homography.

The pitch is flat, so a 3x3 homography maps image pixels (of the reference
frame) to pitch coordinates in metres. It is estimated from at least four
pitch landmarks whose pixel positions are entered by the user and whose real
positions are known from the standard pitch markings.

Pitch coordinate system used throughout the project:
    x: 0 .. length (metres), from the left goal line to the right goal line
       as seen from the main camera
    y: 0 .. width  (metres), from the far touchline (top of the image)
       to the near touchline (bottom of the image)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import config

CENTRE_CIRCLE_R = 9.15
PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16
PENALTY_SPOT = 11.0


class CalibrationError(Exception):
    pass


def pitch_landmarks(length: float = config.PITCH_LENGTH_M, width: float = config.PITCH_WIDTH_M) -> dict[str, tuple[float, float]]:
    """Named pitch landmarks and their coordinates in metres."""
    L, W = float(length), float(width)
    cy = W / 2
    marks = {
        "Corner, far left": (0.0, 0.0),
        "Corner, far right": (L, 0.0),
        "Corner, near left": (0.0, W),
        "Corner, near right": (L, W),
        "Halfway line, far touchline": (L / 2, 0.0),
        "Halfway line, near touchline": (L / 2, W),
        "Centre spot": (L / 2, cy),
        "Centre circle, far point": (L / 2, cy - CENTRE_CIRCLE_R),
        "Centre circle, near point": (L / 2, cy + CENTRE_CIRCLE_R),
        "Centre circle, left point": (L / 2 - CENTRE_CIRCLE_R, cy),
        "Centre circle, right point": (L / 2 + CENTRE_CIRCLE_R, cy),
    }
    for side, gx, sign in (("Left", 0.0, 1.0), ("Right", L, -1.0)):
        marks[f"{side} penalty area, far corner on goal line"] = (gx, cy - PENALTY_HALF_WIDTH)
        marks[f"{side} penalty area, far front corner"] = (gx + sign * PENALTY_DEPTH, cy - PENALTY_HALF_WIDTH)
        marks[f"{side} penalty area, near front corner"] = (gx + sign * PENALTY_DEPTH, cy + PENALTY_HALF_WIDTH)
        marks[f"{side} penalty area, near corner on goal line"] = (gx, cy + PENALTY_HALF_WIDTH)
        marks[f"{side} goal area, far corner on goal line"] = (gx, cy - GOAL_AREA_HALF_WIDTH)
        marks[f"{side} goal area, far front corner"] = (gx + sign * GOAL_AREA_DEPTH, cy - GOAL_AREA_HALF_WIDTH)
        marks[f"{side} goal area, near front corner"] = (gx + sign * GOAL_AREA_DEPTH, cy + GOAL_AREA_HALF_WIDTH)
        marks[f"{side} goal area, near corner on goal line"] = (gx, cy + GOAL_AREA_HALF_WIDTH)
        marks[f"{side} penalty spot"] = (gx + sign * PENALTY_SPOT, cy)
    return marks


@dataclass
class Calibration:
    landmark_names: list[str]
    image_points: list[list[float]]
    pitch_points: list[list[float]]
    pitch_length: float
    pitch_width: float
    homography: list[list[float]]          # reference image pixels -> pitch metres
    reprojection_error_m: float
    created: str

    @property
    def H(self) -> np.ndarray:
        return np.asarray(self.homography, dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _polygon_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def compute_calibration(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    landmark_names: list[str],
    length: float = config.PITCH_LENGTH_M,
    width: float = config.PITCH_WIDTH_M,
) -> Calibration:
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pitch = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
    if len(img) < 4 or len(img) != len(pitch):
        raise CalibrationError("At least four landmark pairs are needed.")
    if len(set(landmark_names)) != len(landmark_names):
        raise CalibrationError("Each landmark can only be used once.")
    if _polygon_area(pitch) < 20.0:
        raise CalibrationError("The chosen landmarks lie almost on one line. Pick points that span an area of the pitch.")
    if _polygon_area(img) < 400.0:
        raise CalibrationError("The image points are too close together or almost on one line.")
    method = 0 if len(img) == 4 else cv2.RANSAC
    h, _ = cv2.findHomography(img, pitch, method, 1.0)
    if h is None or not np.all(np.isfinite(h)):
        raise CalibrationError("The homography could not be computed from these points.")
    projected = cv2.perspectiveTransform(img.reshape(-1, 1, 2), h).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected - pitch, axis=1)))
    return Calibration(
        landmark_names=list(landmark_names),
        image_points=img.tolist(),
        pitch_points=pitch.tolist(),
        pitch_length=float(length),
        pitch_width=float(width),
        homography=(h / h[2, 2]).tolist(),
        reprojection_error_m=round(error, 3),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def image_to_pitch(points: np.ndarray, calibration: Calibration) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), calibration.H).reshape(-1, 2)


def pitch_line_polylines(length: float = config.PITCH_LENGTH_M, width: float = config.PITCH_WIDTH_M) -> list[np.ndarray]:
    """Pitch markings as polylines in metres."""
    L, W = float(length), float(width)
    cy = W / 2
    lines = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]),
        np.array([[L / 2, 0], [L / 2, W]]),
    ]
    angles = np.linspace(0, 2 * np.pi, 73)
    lines.append(np.stack([L / 2 + CENTRE_CIRCLE_R * np.cos(angles), cy + CENTRE_CIRCLE_R * np.sin(angles)], axis=1))
    for gx, sign in ((0.0, 1.0), (L, -1.0)):
        px = gx + sign * PENALTY_DEPTH
        lines.append(np.array([[gx, cy - PENALTY_HALF_WIDTH], [px, cy - PENALTY_HALF_WIDTH],
                               [px, cy + PENALTY_HALF_WIDTH], [gx, cy + PENALTY_HALF_WIDTH]]))
        ga = gx + sign * GOAL_AREA_DEPTH
        lines.append(np.array([[gx, cy - GOAL_AREA_HALF_WIDTH], [ga, cy - GOAL_AREA_HALF_WIDTH],
                               [ga, cy + GOAL_AREA_HALF_WIDTH], [gx, cy + GOAL_AREA_HALF_WIDTH]]))
        # Penalty arc: the part of the circle around the penalty spot outside the area.
        spot = gx + sign * PENALTY_SPOT
        arc = np.linspace(-np.pi, np.pi, 145)
        xs, ys = spot + CENTRE_CIRCLE_R * np.cos(arc), cy + CENTRE_CIRCLE_R * np.sin(arc)
        outside = (xs > px) if sign > 0 else (xs < px)
        if outside.any():
            lines.append(np.stack([xs[outside], ys[outside]], axis=1))
    return lines


def draw_projected_pitch(image_bgr: np.ndarray, calibration: Calibration, colour=(0, 255, 255)) -> np.ndarray:
    """Draw the pitch markings projected into the image to check the calibration visually."""
    out = image_bgr.copy()
    inv = np.linalg.inv(calibration.H)
    # A homography is only defined up to scale. Flip the sign if needed so that the
    # entered landmarks (which are in front of the camera) get a positive w.
    known = np.hstack([np.asarray(calibration.pitch_points, dtype=np.float64), np.ones((len(calibration.pitch_points), 1))])
    if np.median(known @ inv[2]) < 0:
        inv = -inv
    h_img, w_img = out.shape[:2]
    for line in pitch_line_polylines(calibration.pitch_length, calibration.pitch_width):
        dense = []
        for a, b in zip(line[:-1], line[1:]):  # densify so curved projection looks right
            for t in np.linspace(0, 1, 20, endpoint=False):
                dense.append(a + (b - a) * t)
        dense.append(line[-1])
        pts = np.asarray(dense, dtype=np.float64)
        homog = np.hstack([pts, np.ones((len(pts), 1))]) @ inv.T
        valid = homog[:, 2] > 1e-9
        proj = homog[:, :2] / np.where(valid, homog[:, 2], 1.0)[:, None]
        valid &= (np.abs(proj[:, 0]) < 4 * w_img) & (np.abs(proj[:, 1]) < 4 * h_img)
        for i in range(1, len(proj)):
            if valid[i] and valid[i - 1]:
                cv2.line(out, tuple(int(v) for v in proj[i - 1]), tuple(int(v) for v in proj[i]), colour, 2, cv2.LINE_AA)
    for x, y in calibration.image_points:
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 255), 2, cv2.LINE_AA)
    return out
