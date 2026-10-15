"""Player tracking with ByteTrack.

ByteTrack (Zhang et al., 2022) links detections from consecutive frames into
tracks. For every frame it:
  1. predicts where each existing track should be with a Kalman filter,
  2. matches high-confidence detections to the predictions by IoU (Hungarian
     algorithm),
  3. matches the remaining tracks to low-confidence detections, which keeps
     players alive through partial occlusion,
  4. starts new tracks from unmatched high-confidence detections and deletes
     tracks that have been lost for longer than the track buffer.

The implementation used here is the one shipped inside the ultralytics
package, so no extra dependency is needed. Tracking IDs are arbitrary
numbers: they do NOT correspond to player names or shirt numbers, and an ID
can change after a long occlusion or when two players cross.
"""
from __future__ import annotations

import inspect
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import config
from modules.player_detector import Detections


def _default_bytetrack_cfg() -> dict:
    """Load the ByteTrack defaults that ship with the installed ultralytics version."""
    import ultralytics

    cfg_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class _BoxesView:
    """Minimal object with the attributes BYTETracker.update() reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        out = np.empty_like(self.xyxy)
        out[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        out[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return out

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, index) -> "_BoxesView":
        return _BoxesView(self.xyxy[index], self.conf[index], self.cls[index])


class PlayerTracker:
    """Assigns persistent IDs to player detections across frames."""

    def __init__(
        self,
        effective_fps: float,
        confidence: float = config.DEFAULT_CONFIDENCE,
        lost_seconds: float = config.TRACK_LOST_SECONDS,
        low_confidence: float = config.TRACKER_LOW_CONFIDENCE,
    ):
        from ultralytics.trackers.byte_tracker import BYTETracker

        cfg = _default_bytetrack_cfg()
        cfg.update(
            {
                "tracker_type": "bytetrack",
                "track_high_thresh": float(confidence),
                "track_low_thresh": float(min(low_confidence, confidence)),
                "new_track_thresh": float(min(0.9, confidence + 0.05)),
                # buffer is counted in processed frames
                "track_buffer": max(1, int(round(lost_seconds * effective_fps))),
                "match_thresh": config.TRACKER_MATCH_THRESHOLD,
                "fuse_score": True,
            }
        )
        self.settings = cfg
        args = SimpleNamespace(**cfg)
        # Older ultralytics versions take a frame_rate argument that scales track_buffer;
        # 30 keeps the buffer equal to the number of processed frames.
        if "frame_rate" in inspect.signature(BYTETracker.__init__).parameters:
            self._tracker = BYTETracker(args, frame_rate=30)
        else:
            self._tracker = BYTETracker(args)
        self._tracker.reset()  # IDs start again at 1 for each video

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> Detections:
        """Feed one frame of detections; return the tracked detections with IDs.

        Detections passed in should include low-confidence boxes (down to the
        tracker's low threshold) because ByteTrack uses them in its second
        association step. Only tracks that were matched in this frame are returned.
        """
        boxes = _BoxesView(detections.xyxy, detections.confidence, detections.class_id)
        tracks = self._tracker.update(boxes, frame)
        if tracks is None or len(tracks) == 0:
            return Detections()
        tracks = np.asarray(tracks, dtype=np.float32)
        # columns: x1, y1, x2, y2, track_id, score, cls, detection_index
        return Detections(
            xyxy=tracks[:, :4].astype(np.float32),
            confidence=tracks[:, 5].astype(np.float32),
            class_id=tracks[:, 6].astype(np.int32),
            tracker_id=tracks[:, 4].astype(np.int32),
        )


class TrailBuffer:
    """Keeps the most recent positions of each track for drawing movement trails."""

    def __init__(self, max_length: int = config.DEFAULT_TRAIL_LENGTH):
        self.max_length = max(1, int(max_length))
        self._points: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_length))
        self._last_seen: dict[int, int] = {}

    def add(self, track_id: int, point: tuple[float, float], frame_number: int) -> None:
        self._points[int(track_id)].append((float(point[0]), float(point[1])))
        self._last_seen[int(track_id)] = frame_number

    def get(self, track_id: int) -> list[tuple[float, float]]:
        return list(self._points.get(int(track_id), []))

    def prune(self, frame_number: int, max_age: int) -> None:
        """Forget tracks that have not been seen for `max_age` processed frames."""
        stale = [tid for tid, seen in self._last_seen.items() if frame_number - seen > max_age]
        for tid in stale:
            self._points.pop(tid, None)
            self._last_seen.pop(tid, None)

    def active_ids(self) -> list[int]:
        return list(self._points.keys())
