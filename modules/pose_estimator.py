"""Optional 2D pose estimation for a single tracked player.

The project specification lists MediaPipe for pose estimation. Current
MediaPipe releases install `opencv-contrib-python`, which conflicts with the
`opencv-python` package used by ultralytics, and need a separately downloaded
model file. To keep the installation reliable this module uses the Ultralytics
YOLO pose model instead (17 COCO body keypoints), which runs with the packages
that are already installed. MediaPipe remains a documented alternative.

The player is cropped from the frame, enlarged and passed to the pose model.
Angles are measured in the 2D image, so they depend on the camera angle and are
only indicative. Broadcast footage often shows players too small for reliable
keypoints; low-confidence keypoints are not drawn or used.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

import config

POSE_MODEL = config.POSE_MODEL
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
MIN_KEYPOINT_CONF = 0.5


@dataclass
class PoseResult:
    image_rgb: np.ndarray
    keypoints: pd.DataFrame
    angles: dict[str, float]
    message: str


def _angle(a, b, c) -> float:
    """Angle ABC in degrees."""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class PoseEstimator:
    def __init__(self, model_name: str = POSE_MODEL):
        from modules.player_detector import load_yolo

        self.model = load_yolo(model_name)

    def estimate(self, frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> PoseResult:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        H, W = frame_bgr.shape[:2]
        pad_x, pad_y = 0.35 * w, 0.15 * h
        cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        cx2, cy2 = int(min(W, x2 + pad_x)), int(min(H, y2 + pad_y))
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            raise ValueError("The selected box is outside the frame.")
        scale = max(1.0, 320.0 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        result = self.model.predict(crop, conf=0.25, imgsz=320, device=config.DEVICE, verbose=False)[0]
        canvas = crop.copy()
        empty = pd.DataFrame(columns=["keypoint", "x", "y", "confidence"])
        if result.keypoints is None or len(result.keypoints) == 0:
            return PoseResult(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), empty, {},
                              f"No body pose was found. The player is {int(h)} px tall in the frame; "
                              "very small players often cannot be analysed.")
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(areas))
        xy = result.keypoints.xy.cpu().numpy()[best]
        conf = result.keypoints.conf.cpu().numpy()[best] if result.keypoints.conf is not None else np.ones(len(xy))
        ok = conf >= MIN_KEYPOINT_CONF

        for a, b in SKELETON:
            if ok[a] and ok[b]:
                cv2.line(canvas, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (235, 152, 42), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(xy):
            if ok[i]:
                cv2.circle(canvas, (int(px), int(py)), 3, (40, 200, 250), -1, cv2.LINE_AA)

        angles = {}
        for name, (a, b, c) in {"left knee": (11, 13, 15), "right knee": (12, 14, 16),
                                "left hip": (5, 11, 13), "right hip": (6, 12, 14),
                                "left elbow": (5, 7, 9), "right elbow": (6, 8, 10)}.items():
            if ok[a] and ok[b] and ok[c]:
                angles[name] = round(_angle(xy[a], xy[b], xy[c]), 1)
        if ok[5] and ok[6] and ok[11] and ok[12]:
            shoulders, hips = (xy[5] + xy[6]) / 2, (xy[11] + xy[12]) / 2
            dx, dy = shoulders - hips
            angles["trunk lean from vertical"] = round(float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))), 1)

        table = pd.DataFrame({
            "keypoint": KEYPOINT_NAMES[: len(xy)],
            "x": np.round(xy[:, 0] / scale + cx1, 1),
            "y": np.round(xy[:, 1] / scale + cy1, 1),
            "confidence": np.round(conf, 2),
        })
        message = f"{int(ok.sum())} of {len(xy)} keypoints have confidence of at least {MIN_KEYPOINT_CONF}."
        return PoseResult(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), table, angles, message)
