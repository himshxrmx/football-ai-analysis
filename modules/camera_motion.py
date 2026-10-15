"""Camera motion compensation.

Broadcast cameras pan and zoom to follow the ball. Without compensation a
player standing still would appear to move across the image, which would
corrupt heatmaps and distance estimates.

For every processed frame this module estimates a homography H that maps
pixel coordinates of the current frame to pixel coordinates of the first
frame:
  1. Corner points (Shi-Tomasi "good features to track") are found on a
     downscaled grey image of the previous frame. Only the grass and its
     markings are used: player boxes are masked out because players move
     independently of the camera, and broadcast overlays (score boxes, logos)
     are excluded because they do not move with the pitch.
  2. The points are followed into the current frame with pyramidal
     Lucas-Kanade optical flow.
  3. RANSAC fits a homography to the point pairs and rejects outliers.
  4. The frame-to-frame homographies are chained, so every position can be
     expressed in one fixed reference view.

Chaining accumulates small errors (drift), so the result is reliable for
short continuous clips but not across scene cuts or replays.
"""
from __future__ import annotations

import cv2
import numpy as np


def _is_reasonable(h: np.ndarray | None) -> bool:
    """Reject homographies that imply impossible camera motion between two frames."""
    if h is None or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        return False
    h = h / h[2, 2]
    det = np.linalg.det(h[:2, :2])
    if not 0.7 < det < 1.4:
        return False
    if abs(h[2, 0]) > 2e-3 or abs(h[2, 1]) > 2e-3:
        return False
    return True


class CameraMotionEstimator:
    def __init__(self, enabled: bool = True, scale: float = 0.5, max_corners: int = 400, min_points: int = 25):
        self.enabled = enabled
        self.scale = scale
        self.max_corners = max_corners
        self.min_points = min_points
        self.h_to_first = np.eye(3, dtype=np.float64)
        self._prev_gray = None
        self._prev_mask = None
        self.steps_ok = 0
        self.steps_failed = 0

    def _prepare(self, frame: np.ndarray, exclude_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
        if green.mean() / 255 > 0.2:
            mask = cv2.dilate(green, np.ones((3, 3), np.uint8), iterations=2)  # include the white lines
        else:
            mask = np.full(gray.shape, 255, dtype=np.uint8)  # no pitch visible: use the whole frame
        h, w = gray.shape
        for x1, y1, x2, y2 in np.asarray(exclude_boxes, dtype=np.float64).reshape(-1, 4) * self.scale:
            pad_x, pad_y = 0.15 * (x2 - x1) + 2, 0.1 * (y2 - y1) + 2
            cv2.rectangle(mask, (int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))),
                          (int(min(w - 1, x2 + pad_x)), int(min(h - 1, y2 + pad_y))), 0, -1)
        return gray, mask

    def _estimate_step(self, gray: np.ndarray) -> np.ndarray | None:
        """Homography mapping the current frame to the previous frame (full resolution)."""
        prev_pts = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=self.max_corners, qualityLevel=0.01,
                                           minDistance=8, mask=self._prev_mask, blockSize=7)
        if prev_pts is None or len(prev_pts) < self.min_points:
            return None
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None,
                                                      winSize=(21, 21), maxLevel=3)
        if cur_pts is None:
            return None
        good = status.ravel() == 1
        if int(good.sum()) < self.min_points:
            return None
        src = cur_pts[good].reshape(-1, 2) / self.scale
        dst = prev_pts[good].reshape(-1, 2) / self.scale
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if _is_reasonable(h) and inliers is not None and int(inliers.sum()) >= self.min_points:
            return h / h[2, 2]
        # A 4-degree-of-freedom similarity is more robust when the homography is unstable.
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if affine is None or inliers is None or int(inliers.sum()) < self.min_points:
            return None
        h = np.vstack([affine, [0.0, 0.0, 1.0]])
        return h if _is_reasonable(h) else None

    def update(self, frame: np.ndarray, exclude_boxes: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Process the next frame. Returns (H_current_to_first, step_ok)."""
        if not self.enabled:
            return np.eye(3), True
        boxes = np.zeros((0, 4)) if exclude_boxes is None else exclude_boxes
        gray, mask = self._prepare(frame, boxes)
        ok = True
        if self._prev_gray is not None:
            step = self._estimate_step(gray)
            ok = step is not None
            if ok:
                self.h_to_first = self.h_to_first @ step
                self.h_to_first /= self.h_to_first[2, 2]
                self.steps_ok += 1
            else:
                self.steps_failed += 1  # assume no camera motion for this step
        self._prev_gray, self._prev_mask = gray, mask
        return self.h_to_first.copy(), ok

    @property
    def success_rate(self) -> float:
        total = self.steps_ok + self.steps_failed
        return 1.0 if total == 0 else self.steps_ok / total


def transform_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), np.asarray(h, dtype=np.float64)).reshape(-1, 2)


def frame_histogram(img: np.ndarray) -> np.ndarray:
    """Normalised hue/saturation histogram used for scene-cut detection."""
    small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def histogram_change(prev_hist: np.ndarray | None, hist: np.ndarray) -> float:
    """1 - correlation of two frame histograms. Values above about 0.5 indicate a scene cut."""
    if prev_hist is None:
        return 0.0
    return float(1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
