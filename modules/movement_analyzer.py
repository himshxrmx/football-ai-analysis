"""Movement analysis: positions, distances, speeds, zones and summary statistics.

Two coordinate spaces are supported:

* "pitch": metres on the pitch. Only available after a camera-to-pitch
  calibration. Distances and speeds are physical estimates.
* "image": pixels of the reference frame (after camera motion compensation).
  Values are only comparable within the same video and are NOT metres.

Positions are the bottom-centre of each bounding box (the feet). They are
smoothed with a centred rolling mean before distances are computed, because
box jitter would otherwise add distance that the player never covered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cv2

import config
from modules.pitch_calibration import image_to_pitch
from modules.run_store import RunData
from modules.team_classifier import assign_colour_groups

PITCH_MARGIN_M = 3.0


def space_for(run: RunData) -> tuple[str, str]:
    """(space, unit) used for movement metrics of this run."""
    return ("pitch", "m") if run.is_calibrated else ("image", "px")


def positions(run: RunData) -> pd.DataFrame:
    """Tracks with `pos_x`/`pos_y` in the run's analysis space.

    In pitch space, positions projected far outside the pitch (usually tracking
    or calibration errors) are dropped.
    """
    df = run.tracks.copy()
    if df.empty:
        return df.assign(pos_x=pd.Series(dtype=float), pos_y=pd.Series(dtype=float))
    df = df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    if run.is_calibrated:
        cal = run.calibration
        pitch = image_to_pitch(df[["ref_x", "ref_y"]].to_numpy(), cal)
        df["pos_x"], df["pos_y"] = pitch[:, 0], pitch[:, 1]
        inside = (
            df["pos_x"].between(-PITCH_MARGIN_M, cal.pitch_length + PITCH_MARGIN_M)
            & df["pos_y"].between(-PITCH_MARGIN_M, cal.pitch_width + PITCH_MARGIN_M)
        )
        df = df[inside].reset_index(drop=True)
    else:
        df["pos_x"], df["pos_y"] = df["ref_x"], df["ref_y"]
    return df


def _moving_average(values: pd.Series, window: int) -> pd.Series:
    """Centred moving average that keeps straight-line motion unchanged at the ends.

    The series is padded with point reflections (2 * end - value) before
    averaging. A plain rolling mean with shrinking windows would pull the first
    and last points inwards and shorten every track by about two frames of movement.
    """
    v = values.to_numpy(dtype=float)
    half = min(window // 2, len(v) - 1)
    if half < 1:
        return values.astype(float)
    left = 2 * v[0] - v[1:half + 1][::-1]
    right = 2 * v[-1] - v[-half - 1:-1][::-1]
    padded = np.concatenate([left, v, right])
    kernel = np.full(2 * half + 1, 1.0 / (2 * half + 1))
    return pd.Series(np.convolve(padded, kernel, mode="valid"), index=values.index)


def smooth(df: pd.DataFrame, window: int = config.SMOOTHING_WINDOW) -> pd.DataFrame:
    """Add `sx`/`sy`: per-track centred moving average of the position."""
    if df.empty:
        return df.assign(sx=pd.Series(dtype=float), sy=pd.Series(dtype=float))
    out = df.copy()
    grouped = out.groupby("track_id", sort=False)
    out["sx"] = grouped["pos_x"].transform(lambda s: _moving_average(s, window))
    out["sy"] = grouped["pos_y"].transform(lambda s: _moving_average(s, window))
    return out


def add_steps(df: pd.DataFrame, stride: int, metric: bool) -> pd.DataFrame:
    """Add per-row `step` (distance since previous observation), `dt` and `speed`.

    Steps across gaps longer than MAX_GAP_FRAMES processed frames are not counted,
    and in metric space steps faster than MAX_HUMAN_SPEED_MPS are rejected.
    """
    if df.empty:
        return df.assign(step=pd.Series(dtype=float), dt=pd.Series(dtype=float), speed=pd.Series(dtype=float),
                         valid_step=pd.Series(dtype=bool), rejected_step=pd.Series(dtype=bool))
    out = df.sort_values(["track_id", "frame_idx"]).copy()
    g = out.groupby("track_id", sort=False)
    dx = g["sx"].diff()
    dy = g["sy"].diff()
    gap = g["frame_idx"].diff() / max(1, stride)
    out["dt"] = g["time_s"].diff()
    out["step"] = np.hypot(dx, dy)
    connected = gap.notna() & (gap <= config.MAX_GAP_FRAMES) & (out["dt"] > 0)
    out["speed"] = np.where(connected, out["step"] / out["dt"].where(out["dt"] > 0), np.nan)
    rejected = pd.Series(False, index=out.index)
    if metric:
        rejected = connected & (out["speed"] > config.MAX_HUMAN_SPEED_MPS)
    out["valid_step"] = connected & ~rejected
    out["rejected_step"] = rejected
    out.loc[~out["valid_step"], ["step", "speed"]] = np.nan
    return out


def _bursts(times: np.ndarray, speeds: np.ndarray, threshold: float, min_seconds: float) -> int:
    """Count continuous runs above `threshold` lasting at least `min_seconds`."""
    count, start = 0, None
    for t, v in zip(times, speeds):
        if np.isfinite(v) and v >= threshold:
            start = t if start is None else start
        else:
            if start is not None and t - start >= min_seconds:
                count += 1
            start = None
    if start is not None and len(times) and times[-1] - start >= min_seconds:
        count += 1
    return count


def _hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def movement_table(run: RunData) -> pd.DataFrame:
    """Smoothed positions with step distances and speeds for every tracked row."""
    space, _ = space_for(run)
    return add_steps(smooth(positions(run)), run.stride, metric=(space == "pitch"))


def player_metrics(run: RunData, steps: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row of metrics per track ID."""
    space, unit = space_for(run)
    metric = space == "pitch"
    steps = movement_table(run) if steps is None else steps
    columns = ["track_id", "role", "colour_group", "stable", "first_seen_s", "last_seen_s", "frames_seen",
               "visible_s", "visibility_pct", "distance", "avg_speed", "peak_speed", "high_speed_distance",
               "sprints", "area_covered", "centroid_x", "centroid_y", "mean_confidence", "rejected_steps", "unit"]
    if steps.empty:
        text = {"role", "colour_group", "unit"}
        dtypes = {c: ("object" if c in text else "bool" if c == "stable" else "int64" if c == "track_id" else "float64")
                  for c in columns}
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})

    dt_frame = 1.0 / run.effective_fps
    total_frames = max(1, run.frames_processed)
    rows = []
    for track_id, grp in steps.groupby("track_id", sort=True):
        grp = grp.sort_values("frame_idx")
        valid = grp[grp["valid_step"]]
        moving_time = float(valid["dt"].sum())
        distance = float(valid["step"].sum())
        speeds = valid["speed"].to_numpy(dtype=float)
        frames_seen = int(len(grp))
        visible_s = frames_seen * dt_frame
        rows.append({
            "track_id": int(track_id),
            "role": grp["role"].mode().iat[0] if "role" in grp and not grp["role"].isna().all() else "person",
            "stable": visible_s >= config.MIN_TRACK_SECONDS,
            "first_seen_s": round(float(grp["time_s"].min()), 2),
            "last_seen_s": round(float(grp["time_s"].max()), 2),
            "frames_seen": frames_seen,
            "visible_s": round(visible_s, 2),
            "visibility_pct": round(100.0 * frames_seen / total_frames, 1),
            "distance": round(distance, 2),
            "avg_speed": round(distance / moving_time, 3) if moving_time > 0 else np.nan,
            "peak_speed": round(float(np.nanpercentile(speeds, 95)), 3) if len(speeds) >= 5 else np.nan,
            "high_speed_distance": round(float(valid.loc[valid["speed"] >= config.HIGH_SPEED_THRESHOLD_MPS, "step"].sum()), 2) if metric else np.nan,
            "sprints": _bursts(valid["time_s"].to_numpy(), speeds, config.SPRINT_THRESHOLD_MPS, config.MIN_BURST_SECONDS) if metric else np.nan,
            "area_covered": round(_hull_area(grp[["sx", "sy"]].to_numpy()), 1),
            "centroid_x": round(float(grp["sx"].mean()), 2),
            "centroid_y": round(float(grp["sy"].mean()), 2),
            "mean_confidence": round(float(grp["confidence"].mean()), 3),
            "rejected_steps": int(grp["rejected_step"].sum()),
            "unit": unit,
        })
    table = pd.DataFrame(rows)
    groups = assign_colour_groups(run.tracks, eligible_ids=table.loc[table["stable"], "track_id"].tolist())
    if not groups.empty:
        table = table.merge(groups[["track_id", "colour_group"]], on="track_id", how="left")
    else:
        table["colour_group"] = np.nan
    table["colour_group"] = table["colour_group"].fillna("Unassigned")
    return table[columns]


def zone_shares(run: RunData, steps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of observations per third (along the length) and channel (across the width).

    In pitch space thirds refer to the pitch. In image space they refer to the
    reference camera view and are only a rough indication.
    """
    steps = movement_table(run) if steps is None else steps
    if steps.empty:
        return pd.DataFrame(columns=["track_id", "left_third", "middle_third", "right_third",
                                     "far_channel", "central_channel", "near_channel"])
    if run.is_calibrated:
        length, width = run.calibration.pitch_length, run.calibration.pitch_width
    else:
        length, width = float(run.meta["video"]["width"]), float(run.meta["video"]["height"])
    x = steps["sx"].clip(0, length)
    y = steps["sy"].clip(0, width)
    third = pd.cut(x, [-np.inf, length / 3, 2 * length / 3, np.inf], labels=["left_third", "middle_third", "right_third"])
    channel = pd.cut(y, [-np.inf, width / 3, 2 * width / 3, np.inf], labels=["far_channel", "central_channel", "near_channel"])
    a = pd.crosstab(steps["track_id"], third, normalize="index")
    b = pd.crosstab(steps["track_id"], channel, normalize="index")
    out = (a.join(b, how="outer") * 100).round(1).fillna(0.0)
    for col in ["left_third", "middle_third", "right_third", "far_channel", "central_channel", "near_channel"]:
        if col not in out:
            out[col] = 0.0
    return out.reset_index()


def overview(run: RunData, metrics: pd.DataFrame | None = None) -> dict:
    """Summary numbers for the Overview page."""
    frames = run.frames
    tracks = run.tracks
    metrics = player_metrics(run) if metrics is None else metrics
    processed = run.frames_processed
    no_detection = int((frames["tracked_players"] == 0).sum()) if not frames.empty else 0
    return {
        "frames_processed": processed,
        "duration_processed_s": run.meta["processing"]["duration_processed_s"],
        "unique_track_ids": int(tracks["track_id"].nunique()) if not tracks.empty else 0,
        "stable_tracks": int(metrics["stable"].sum()) if not metrics.empty else 0,
        "mean_tracked_per_frame": round(float(frames["tracked_players"].mean()), 1) if not frames.empty else 0.0,
        "max_tracked_in_frame": int(frames["tracked_players"].max()) if not frames.empty else 0,
        "frames_without_players": no_detection,
        "frames_without_players_pct": round(100.0 * no_detection / max(1, processed), 1),
        "mean_confidence": round(float(tracks["confidence"].mean()), 3) if not tracks.empty else None,
        "ball_detection_pct": round(100.0 * frames["ball_detected"].mean(), 1) if not frames.empty else 0.0,
        "camera_success_pct": round(100.0 * run.meta["processing"]["camera_success_rate"], 1),
        "scene_cuts": run.meta["processing"]["scene_cuts"],
    }


def speed_series(steps: pd.DataFrame, track_id: int, window_s: float = 1.0, fps: float = 12.5) -> pd.DataFrame:
    """Smoothed speed over time for one track (rolling mean over about `window_s`)."""
    grp = steps[(steps["track_id"] == track_id) & steps["valid_step"]].sort_values("time_s")
    if grp.empty:
        return pd.DataFrame(columns=["time_s", "speed"])
    window = max(1, int(round(window_s * fps)))
    return pd.DataFrame({"time_s": grp["time_s"], "speed": grp["speed"].rolling(window, min_periods=1).mean()})


def group_spread(steps: pd.DataFrame, metrics: pd.DataFrame, min_players: int = 3) -> pd.DataFrame:
    """Average distance of each colour group's players to the group centroid, per frame."""
    if steps.empty or metrics.empty:
        return pd.DataFrame(columns=["frame_idx", "time_s", "colour_group", "spread", "players"])
    lookup = metrics.set_index("track_id")["colour_group"]
    df = steps.assign(colour_group=steps["track_id"].map(lookup))
    df = df[df["colour_group"].isin(["Group 1", "Group 2"])]
    rows = []
    for (frame_idx, group), grp in df.groupby(["frame_idx", "colour_group"]):
        if len(grp) < min_players:
            continue
        cx, cy = grp["sx"].mean(), grp["sy"].mean()
        rows.append({"frame_idx": frame_idx, "time_s": grp["time_s"].iat[0], "colour_group": group,
                     "spread": float(np.hypot(grp["sx"] - cx, grp["sy"] - cy).mean()), "players": len(grp)})
    return pd.DataFrame(rows, columns=["frame_idx", "time_s", "colour_group", "spread", "players"])
