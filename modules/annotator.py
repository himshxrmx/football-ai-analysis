"""Drawing of boxes, IDs, confidence scores and movement trails on frames."""
from __future__ import annotations

import cv2
import numpy as np

from modules.player_detector import Detections

# Distinct, high-contrast colours (BGR) for track IDs.
PALETTE_BGR = [
    (63, 208, 244),   # yellow
    (226, 173, 93),   # blue
    (78, 152, 235),   # orange
    (141, 214, 88),   # green
    (99, 112, 236),   # red
    (176, 201, 72),   # teal
    (177, 183, 245),  # pink
    (241, 214, 174),  # light blue
    (113, 196, 248),  # amber
    (170, 224, 130),  # light green
    (152, 152, 229),  # salmon
    (250, 250, 250),  # white
]
ROLE_COLOURS_BGR = {"referee": (40, 40, 40), "goalkeeper": (0, 215, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_for_id(track_id: int) -> tuple[int, int, int]:
    if track_id is None or track_id < 0:
        return (180, 180, 180)
    return PALETTE_BGR[int(track_id) % len(PALETTE_BGR)]


def colour_for_id_hex(track_id: int) -> str:
    b, g, r = colour_for_id(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"


def _scale_for(frame: np.ndarray) -> float:
    return max(0.4, min(1.2, frame.shape[1] / 1600))


def _label(img: np.ndarray, text: str, x: int, y: int, colour, scale: float) -> None:
    thickness = 1 if scale < 0.7 else 2
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), colour, -1)
    luminance = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    text_colour = (20, 20, 20) if luminance > 140 else (245, 245, 245)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, text_colour, thickness, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    players: Detections,
    ball: Detections | None = None,
    roles: list[str] | None = None,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    show_confidence: bool = True,
    hud_text: str | None = None,
) -> np.ndarray:
    """Return a copy of `frame` with annotations drawn on it."""
    img = frame.copy()
    scale = _scale_for(img)
    thickness = max(1, int(round(2 * scale)))

    # Trails first so boxes are drawn on top.
    if trails:
        for track_id, points in trails.items():
            if len(points) < 2:
                continue
            colour = colour_for_id(track_id)
            pts = np.asarray(points, dtype=np.float32)
            for i in range(1, len(pts)):
                weight = i / len(pts)
                t = max(1, int(round(thickness * (0.5 + weight))))
                p1 = tuple(int(v) for v in pts[i - 1])
                p2 = tuple(int(v) for v in pts[i])
                cv2.line(img, p1, p2, colour, t, cv2.LINE_AA)

    for i in range(len(players)):
        x1, y1, x2, y2 = (int(v) for v in players.xyxy[i])
        track_id = int(players.tracker_id[i])
        role = roles[i] if roles else "person"
        colour = ROLE_COLOURS_BGR.get(role, colour_for_id(track_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
        cx = (x1 + x2) // 2
        cv2.ellipse(img, (cx, y2), (max(4, (x2 - x1) // 2), max(2, (x2 - x1) // 6)), 0, 0, 360, colour, thickness, cv2.LINE_AA)
        parts = []
        if track_id >= 0:
            parts.append(f"ID {track_id}")
        if role in ("referee", "goalkeeper"):
            parts.append(role[:3].upper())
        if show_confidence:
            parts.append(f"{float(players.confidence[i]):.2f}")
        if parts:
            _label(img, " ".join(parts), x1, y1 - 2, colour, 0.45 * scale + 0.1)

    if ball is not None:
        for i in range(len(ball)):
            x1, y1, x2, y2 = ball.xyxy[i]
            cx, top = int((x1 + x2) / 2), int(y1) - 4
            size = max(6, int(10 * scale))
            tri = np.array([[cx, top], [cx - size, top - size * 2], [cx + size, top - size * 2]], dtype=np.int32)
            cv2.fillPoly(img, [tri], (255, 255, 255))
            cv2.polylines(img, [tri], True, (30, 30, 30), 1, cv2.LINE_AA)

    if hud_text:
        hud_scale = 0.5 * scale + 0.15
        (tw, th), base = cv2.getTextSize(hud_text, FONT, hud_scale, 1)
        h = img.shape[0]
        cv2.rectangle(img, (8, h - th - 18), (8 + tw + 12, h - 6), (25, 25, 25), -1)
        cv2.putText(img, hud_text, (14, h - 12), FONT, hud_scale, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
