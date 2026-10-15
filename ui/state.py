"""Cached access to saved analyses and the models used by the interface."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modules import movement_analyzer as ma
from modules.player_detector import PlayerDetector
from modules.run_store import RunData, calibration_path, list_runs, load_run

RUN_KEY = "active_run_dir"


def _stamp(run_dir: str) -> tuple:
    """Changes whenever the run or its calibration changes, so caches are refreshed."""
    meta = Path(run_dir) / "meta.json"
    cal = calibration_path(Path(run_dir))
    return (meta.stat().st_mtime if meta.exists() else 0, cal.stat().st_mtime if cal.exists() else 0)


@st.cache_data(show_spinner=False, max_entries=8)
def _load(run_dir: str, stamp: tuple) -> RunData:
    return load_run(run_dir)


@st.cache_data(show_spinner="Computing movement statistics...", max_entries=8)
def _bundle(run_dir: str, stamp: tuple) -> dict:
    run = load_run(run_dir)
    steps = ma.movement_table(run)
    metrics = ma.player_metrics(run, steps)
    return {
        "steps": steps,
        "metrics": metrics,
        "zones": ma.zone_shares(run, steps),
        "overview": ma.overview(run, metrics),
        "spread": ma.group_spread(steps, metrics),
    }


@st.cache_resource(show_spinner="Loading the detection model...")
def detector(model_name: str) -> PlayerDetector:
    return PlayerDetector(model_name)


def runs() -> list[dict]:
    return list_runs()


def active_run_dir() -> str | None:
    run_dir = st.session_state.get(RUN_KEY)
    if run_dir and (Path(run_dir) / "meta.json").exists():
        return run_dir
    available = runs()
    if available:
        st.session_state[RUN_KEY] = available[0]["run_dir"]
        return available[0]["run_dir"]
    return None


def set_active_run(run_dir: str | Path) -> None:
    st.session_state[RUN_KEY] = str(run_dir)


def current_run() -> RunData | None:
    run_dir = active_run_dir()
    if run_dir is None:
        return None
    return _load(run_dir, _stamp(run_dir))


def bundle(run: RunData) -> dict:
    return _bundle(str(run.run_dir), _stamp(str(run.run_dir)))


def clear_caches() -> None:
    _load.clear()
    _bundle.clear()


def stable_ids(metrics: pd.DataFrame) -> list[int]:
    """Stable track IDs, longest-tracked first.

    Tracks in a colour group come before unassigned ones (often referees or staff),
    so default selections show players.
    """
    if metrics.empty:
        return []
    stable = metrics.loc[metrics["stable"]].assign(unassigned=lambda d: d["colour_group"] == "Unassigned")
    return stable.sort_values(["unassigned", "visible_s"], ascending=[True, False])["track_id"].astype(int).tolist()
