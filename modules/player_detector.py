"""Player detection with a pretrained Ultralytics YOLO model.

Supported models
----------------
1. Official COCO-pretrained YOLO11 weights (default). COCO has 80 classes.
   This project uses two of them:
     * class 0  "person"       -> every visible person (players, goalkeepers,
                                   referees, coaches, spectators)
     * class 32 "sports ball"  -> the ball (often missed when it is small)
   A COCO model cannot tell a player from a referee. The optional pitch
   filter removes people standing off the grass (crowd, bench).

2. An optional football-specific model trained on classes such as
   "player", "goalkeeper", "referee" and "ball". When such a model is loaded
   the class names are read from the weights and mapped to roles
   automatically, so referees are reported separately from players.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config

PERSON_ROLES = ("player", "goalkeeper", "referee", "person")
BALL_NAMES = ("ball", "sports ball", "football", "soccer ball")


class ModelLoadError(Exception):
    """Raised when the detection model cannot be loaded."""


@dataclass
class ClassMap:
    """Which model class ids are people and which are the ball."""

    names: dict[int, str]
    person_ids: list[int]
    ball_ids: list[int]
    roles: dict[int, str]
    model_type: str  # "coco" or "football"

    @property
    def can_separate_referees(self) -> bool:
        return any(role == "referee" for role in self.roles.values())


@dataclass
class Detections:
    """Detections for one frame. Arrays share the same length N."""

    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    class_id: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    tracker_id: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def filter(self, mask: np.ndarray) -> "Detections":
        mask = np.asarray(mask, dtype=bool)
        return Detections(
            xyxy=self.xyxy[mask],
            confidence=self.confidence[mask],
            class_id=self.class_id[mask],
            tracker_id=self.tracker_id[mask],
        )

    @property
    def foot_points(self) -> np.ndarray:
        """Bottom-centre of each box: the point where the player touches the ground."""
        if len(self) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        y = self.xyxy[:, 3]
        return np.stack([x, y], axis=1)


def resolve_class_map(names: dict[int, str]) -> ClassMap:
    """Map model class names to the roles used by this project."""
    lowered = {int(k): str(v).strip().lower() for k, v in names.items()}
    roles: dict[int, str] = {}
    for class_id, name in lowered.items():
        if name in BALL_NAMES:
            roles[class_id] = "ball"
        elif name in ("player", "players"):
            roles[class_id] = "player"
        elif name in ("goalkeeper", "goalie", "keeper"):
            roles[class_id] = "goalkeeper"
        elif name in ("referee", "ref", "linesman"):
            roles[class_id] = "referee"
        elif name == "person":
            roles[class_id] = "person"
    person_ids = sorted(k for k, r in roles.items() if r in PERSON_ROLES)
    ball_ids = sorted(k for k, r in roles.items() if r == "ball")
    model_type = "football" if any(r in ("player", "goalkeeper", "referee") for r in roles.values()) else "coco"
    if not person_ids:
        raise ModelLoadError(
            "The model has no 'person' or 'player' class, so it cannot be used for player detection. "
            f"Classes found: {', '.join(lowered.values())[:300]}"
        )
    return ClassMap(names=dict(names), person_ids=person_ids, ball_ids=ball_ids, roles=roles, model_type=model_type)


def resolve_model_path(model_name: str) -> Path:
    """Official weights live in MODELS_DIR; custom weights may be anywhere."""
    candidate = Path(model_name)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config.MODELS_DIR / model_name


def load_yolo(model_name: str):
    """Load a YOLO model, downloading official weights on first use."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ModelLoadError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from exc

    path = resolve_model_path(model_name)
    is_official = path.name in config.PRETRAINED_MODELS or path.name == config.POSE_MODEL
    if not path.exists() and not is_official:
        raise ModelLoadError(f"Model file not found: {path}")
    try:
        return YOLO(str(path))
    except Exception as exc:
        hint = (
            " The weights are downloaded from GitHub on first use, so an internet connection is needed once."
            if is_official and not path.exists()
            else " The file may be corrupted or incompatible with the installed ultralytics version."
        )
        raise ModelLoadError(f"Could not load model '{path.name}': {exc}.{hint}") from exc


class PlayerDetector:
    """Wraps a YOLO model and returns people and ball detections separately."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL, device: str = config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = load_yolo(model_name)
        self.class_map = resolve_class_map(self.model.names)

    def predict_raw(self, frame: np.ndarray, conf: float, imgsz: int, include_ball: bool = True) -> Detections:
        """Run YOLO on one BGR frame and return every kept detection."""
        if frame is None or frame.size == 0:
            return Detections()
        classes = list(self.class_map.person_ids)
        if include_ball:
            classes += self.class_map.ball_ids
        results = self.model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            classes=classes,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections()
        boxes = boxes.cpu().numpy()
        n = len(boxes)
        return Detections(
            xyxy=boxes.xyxy.astype(np.float32),
            confidence=boxes.conf.astype(np.float32),
            class_id=boxes.cls.astype(np.int32),
            tracker_id=np.full(n, -1, dtype=np.int32),
        )

    def split(self, detections: Detections) -> tuple[Detections, Detections]:
        """Split detections into (people, ball)."""
        if len(detections) == 0:
            return Detections(), Detections()
        is_ball = np.isin(detections.class_id, self.class_map.ball_ids)
        return detections.filter(~is_ball), detections.filter(is_ball)

    def detect(self, frame: np.ndarray, conf: float = config.DEFAULT_CONFIDENCE,
               imgsz: int = config.DEFAULT_IMAGE_SIZE, include_ball: bool = True) -> tuple[Detections, Detections]:
        """Detect people and the ball in a frame."""
        return self.split(self.predict_raw(frame, conf, imgsz, include_ball))

    def role_of(self, class_id: int) -> str:
        return self.class_map.roles.get(int(class_id), "unknown")


# ---------------------------------------------------------------------------
# Pitch filter: drop people whose feet are not on the grass
# ---------------------------------------------------------------------------
PITCH_MASK_SCALE = 0.25


def compute_pitch_mask(frame: np.ndarray, scale: float = PITCH_MASK_SCALE) -> np.ndarray | None:
    """Return a low-resolution mask of the playing area, or None if no pitch is visible.

    The mask is the convex hull of the largest green region, so white lines and
    players standing on the grass are inside it while the stands are outside.
    """
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.15 * green.shape[0] * green.shape[1]:
        return None  # not a wide pitch view (close-up, crowd shot)
    mask = np.zeros_like(green)
    cv2.fillConvexPoly(mask, cv2.convexHull(largest), 255)
    return cv2.dilate(mask, kernel, iterations=2)


def on_pitch_mask(detections: Detections, pitch_mask: np.ndarray | None, scale: float = PITCH_MASK_SCALE) -> np.ndarray:
    """Boolean array: True where the detection's foot point lies inside the pitch mask."""
    if len(detections) == 0:
        return np.zeros((0,), dtype=bool)
    if pitch_mask is None:
        return np.ones(len(detections), dtype=bool)
    feet = detections.foot_points * scale
    h, w = pitch_mask.shape[:2]
    xs = np.clip(feet[:, 0].astype(int), 0, w - 1)
    ys = np.clip(feet[:, 1].astype(int), 0, h - 1)
    return pitch_mask[ys, xs] > 0
