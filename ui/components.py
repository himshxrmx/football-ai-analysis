"""Small reusable interface pieces."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from modules import heatmap_generator as hg
from modules.run_store import RunData

PALETTE_KEY = "series_slots"


def page_header(title: str, description: str | None = None) -> None:
    st.title(title)
    if description:
        st.markdown(f"<p class='note'>{description}</p>", unsafe_allow_html=True)


def run_caption(run: RunData) -> None:
    video = run.meta["video"]
    st.caption(f"Analysis {run.run_id}  |  {video['file_name']}  |  created {run.meta['created']}")


def no_run_message() -> None:
    st.info("No analysis is available yet. Open **Analyse a video** to process a clip. "
            "Results appear on this page once an analysis has finished.")
    st.page_link(st.session_state["pages"]["analyse"], label="Go to Analyse a video")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """items: (label, value, help) rendered as bordered metric cards."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text, border=True)


def fmt(value, digits: int = 1, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    if isinstance(value, int):
        text = f"{value:,}"
    else:
        text = f"{float(value):,.{digits}f}"
    return f"{text} {unit}".strip()


def units(run: RunData) -> dict:
    if run.is_calibrated:
        return {"dist": "m", "speed": "m/s", "area": "m²", "space": "pitch",
                "note": "Distances and speeds are estimates in metres from the pitch calibration."}
    return {"dist": "px", "speed": "px/s", "area": "px²", "space": "image",
            "note": "The video is not calibrated, so distances and speeds are in pixels of the reference camera "
                    "view. They are only comparable within this video and are not metres."}


def unit_note(run: RunData) -> None:
    u = units(run)
    if run.is_calibrated:
        st.caption(u["note"])
    else:
        st.warning(u["note"] + " Use **Pitch calibration** to convert them to metres.")


def series_colours(ids: list[int], key: str) -> dict[int, str]:
    """Stable colours for selected IDs: a colour stays with its ID while it remains selected."""
    store = st.session_state.setdefault(PALETTE_KEY, {}).setdefault(key, {})
    for stale in [i for i in store if i not in ids]:
        store.pop(stale)
    used = set(store.values())
    for i in ids:
        if i not in store:
            free = [s for s in range(hg.MAX_SERIES) if s not in used]
            store[i] = free[0] if free else 0
            used.add(store[i])
    return {i: hg.SERIES[store[i]] for i in ids}


def key_value_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(pd.DataFrame(rows, columns=["Item", "Value"]), hide_index=True, width="stretch")


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
