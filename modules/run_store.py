"""Loading and listing saved analyses from the outputs folder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from modules.pitch_calibration import Calibration


@dataclass
class RunData:
    run_dir: Path
    meta: dict
    tracks: pd.DataFrame
    ball: pd.DataFrame
    frames: pd.DataFrame
    calibration: Calibration | None

    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", self.run_dir.name)

    @property
    def annotated_video(self) -> Path:
        return self.run_dir / "annotated.mp4"

    @property
    def reference_frame(self) -> Path:
        return self.run_dir / "reference_frame.jpg"

    @property
    def effective_fps(self) -> float:
        return float(self.meta["processing"]["effective_fps"])

    @property
    def stride(self) -> int:
        return int(self.meta["settings"]["frame_stride"])

    @property
    def frames_processed(self) -> int:
        return int(self.meta["processing"]["frames_processed"])

    @property
    def video_path(self) -> Path:
        return Path(self.meta["video"]["path"])

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def list_runs(outputs_dir: Path = config.OUTPUTS_DIR) -> list[dict]:
    """Saved analyses, newest first."""
    runs = []
    if not outputs_dir.exists():
        return runs
    for meta_path in outputs_dir.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({
            "run_id": meta.get("run_id", meta_path.parent.name),
            "run_dir": str(meta_path.parent),
            "video": meta.get("video", {}).get("file_name", "unknown"),
            "created": meta.get("created", ""),
            "frames": meta.get("processing", {}).get("frames_processed", 0),
        })
    return sorted(runs, key=lambda r: r["created"], reverse=True)


def calibration_path(run_dir: Path) -> Path:
    return Path(run_dir) / "calibration.json"


def load_run(run_dir: str | Path) -> RunData:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    calibration = None
    cpath = calibration_path(run_dir)
    if cpath.exists():
        try:
            calibration = Calibration.load(cpath)
        except (OSError, json.JSONDecodeError, TypeError):
            calibration = None
    return RunData(
        run_dir=run_dir,
        meta=meta,
        tracks=_read_csv(run_dir / "tracks.csv"),
        ball=_read_csv(run_dir / "ball.csv"),
        frames=_read_csv(run_dir / "frames.csv"),
        calibration=calibration,
    )


def save_calibration(run_dir: str | Path, calibration: Calibration) -> None:
    calibration.save(calibration_path(Path(run_dir)))


def remove_calibration(run_dir: str | Path) -> None:
    path = calibration_path(Path(run_dir))
    if path.exists():
        path.unlink()
