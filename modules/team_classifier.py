"""Experimental grouping of tracks by shirt colour.

For every tracked box the median colour of the torso is measured in the
CIE Lab colour space after removing grass-coloured pixels. Each track then
gets its median shirt colour, and k-means (k = 2) splits the tracks into two
colour groups. Tracks far from both group centres (often referees or
goalkeepers) are left unassigned.

Limitations: the groups are colour clusters, not verified teams. Similar kit
colours, shadows, motion blur and small players reduce the accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


GRASS_DISTANCE = 22.0  # Lab distance below which a pixel is treated as grass


def grass_colour_lab(frame: np.ndarray) -> np.ndarray | None:
    """Median Lab colour of the grass in a frame, or None if little grass is visible.

    The grass colour is measured per frame instead of using a fixed green range,
    so bright green shirts are not mistaken for grass.
    """
    small = cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)) > 0
    if mask.mean() < 0.2:
        return None
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[mask]
    return np.median(lab, axis=0).astype(np.float32)


def jersey_colour_lab(frame: np.ndarray, box: np.ndarray, grass_lab: np.ndarray | None = None) -> np.ndarray | None:
    """Median Lab colour of the upper torso, ignoring pixels close to the grass colour."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    H, W = frame.shape[:2]
    cx1, cx2 = int(max(0, x1 + 0.25 * w)), int(min(W, x2 - 0.25 * w))
    cy1, cy2 = int(max(0, y1 + 0.15 * h)), int(min(H, y1 + 0.5 * h))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    lab = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if grass_lab is not None:
        lab = lab[np.linalg.norm(lab - grass_lab, axis=1) > GRASS_DISTANCE]
    if len(lab) < 8:
        return None
    return np.median(lab, axis=0).astype(np.float32)


def lab_to_hex(lab: np.ndarray) -> str:
    pixel = np.clip(np.asarray(lab, dtype=np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _kmeans(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    cv2.setRNGSeed(42)
    _, labels, centres = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centres


def _cluster(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (labels, centres, ids of the two team clusters).

    Referees, goalkeepers and staff often wear a third colour. k-means with
    k = 3 is tried first: if its smallest cluster is a minority (at most 30% of
    the tracks) and the two large clusters are clearly different, the small
    cluster is treated as "other colours". Otherwise k = 2 is used.
    """
    if len(features) >= 6:
        labels, centres = _kmeans(features, 3)
        sizes = np.bincount(labels, minlength=3)
        order = np.argsort(sizes)
        small, big = int(order[0]), [int(order[1]), int(order[2])]
        if sizes[small] <= 0.3 * len(features) and np.linalg.norm(centres[big[0]] - centres[big[1]]) > 25:
            return labels, centres, big
    labels, centres = _kmeans(features, 2)
    return labels, centres, [0, 1]


def assign_colour_groups(tracks: pd.DataFrame, eligible_ids: list[int] | None = None, min_samples: int = 5) -> pd.DataFrame:
    """Return one row per track: track_id, colour_group, swatch.

    colour_group is "Group 1", "Group 2" or "Unassigned".
    """
    columns = ["track_id", "colour_group", "swatch", "colour_distance"]
    needed = {"track_id", "jersey_l", "jersey_a", "jersey_b"}
    if tracks.empty or not needed.issubset(tracks.columns):
        return pd.DataFrame(columns=columns)
    data = tracks.dropna(subset=["jersey_l", "jersey_a", "jersey_b"])
    if eligible_ids is not None:
        data = data[data["track_id"].isin(eligible_ids)]
    counts = data.groupby("track_id").size()
    keep_ids = counts[counts >= min_samples].index
    data = data[data["track_id"].isin(keep_ids)]
    if data.empty:
        return pd.DataFrame(columns=columns)
    per_track = data.groupby("track_id")[["jersey_l", "jersey_a", "jersey_b"]].median()
    if len(per_track) < 4:
        out = pd.DataFrame({"track_id": per_track.index, "colour_group": "Unassigned"})
        out["swatch"] = [lab_to_hex(v) for v in per_track.values]
        out["colour_distance"] = np.nan
        return out

    # Lightness is down-weighted because shadows change it more than hue.
    weights = np.array([0.5, 1.0, 1.0], dtype=np.float32)
    features = (per_track.values * weights).astype(np.float32)
    labels, centres, team_clusters = _cluster(features)
    distances = np.linalg.norm(features - centres[labels], axis=1)
    team_mask = np.isin(labels, team_clusters)
    median_dist = float(np.median(distances[team_mask])) if team_mask.any() else 0.0
    limit = max(2.5 * median_dist, 12.0)

    # Order the two team clusters by lightness so the naming is stable between runs.
    order = sorted(team_clusters, key=lambda c: centres[c, 0])
    names = {int(order[0]): "Group 1", int(order[1]): "Group 2"}
    groups = [names[int(lbl)] if int(lbl) in names and d <= limit else "Unassigned"
              for lbl, d in zip(labels, distances)]
    return pd.DataFrame(
        {
            "track_id": per_track.index.astype(int),
            "colour_group": groups,
            "swatch": [lab_to_hex(v) for v in per_track.values],
            "colour_distance": np.round(distances, 2),
        }
    )
