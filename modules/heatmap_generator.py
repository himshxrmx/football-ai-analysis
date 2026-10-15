"""Heatmaps, movement paths and pitch drawings (Plotly for the app, Matplotlib for PDF).

Colour rules used in all charts:
* magnitude (heatmap density) uses one blue hue from light to dark,
* identities (players, colour groups) use a fixed categorical order that was
  checked for colour-vision-deficiency separation; at most six at once.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import config  # noqa: E402
from modules.pitch_calibration import pitch_line_polylines  # noqa: E402

# Categorical order validated for colour-vision deficiency (adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAX_SERIES = len(SERIES)
UNASSIGNED = "#898781"
GROUP_COLOURS = {"Group 1": SERIES[0], "Group 2": SERIES[1], "Unassigned": UNASSIGNED}

# Sequential blue ramp (light to dark).
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PITCH_SURFACE = "#eef3ec"
PITCH_LINE = "#8fa58f"

DENSITY_COLORSCALE = [
    [0.0, "rgba(252,252,251,0)"],
    [0.04, "rgba(205,226,251,0.55)"],
    [0.2, BLUE_RAMP[1]],
    [0.4, BLUE_RAMP[2]],
    [0.6, BLUE_RAMP[3]],
    [0.8, BLUE_RAMP[5]],
    [1.0, BLUE_RAMP[6]],
]


def base_layout(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        margin=dict(l=64, r=16, t=48 if title else 36, b=52),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK_SECONDARY),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), automargin=True,
                     title_font=dict(color=INK_SECONDARY, size=12))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), automargin=True,
                     title_font=dict(color=INK_SECONDARY, size=12))
    return fig


def _compact_margins(fig: go.Figure, title: str | None) -> None:
    """Figures without visible axes (pitch, images) need little margin."""
    fig.update_layout(margin=dict(l=8, r=8, t=48 if title else 30, b=8))


# ---------------------------------------------------------------------------
# Pitch (top-down) figures
# ---------------------------------------------------------------------------
def pitch_figure(length: float = config.PITCH_LENGTH_M, width: float = config.PITCH_WIDTH_M,
                 title: str | None = None, height: int = 460) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-2, y0=-2, x1=length + 2, y1=width + 2,
                  fillcolor=PITCH_SURFACE, line=dict(width=0), layer="below")
    for line in pitch_line_polylines(length, width):
        fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=PITCH_LINE, width=1.5)))
    base_layout(fig, title, height)
    _compact_margins(fig, title)
    fig.update_xaxes(range=[-3, length + 3], showgrid=False, zeroline=False, visible=False, constrain="domain")
    fig.update_yaxes(range=[width + 3, -3], showgrid=False, zeroline=False, visible=False,
                     scaleanchor="x", scaleratio=1)
    return fig


def density_grid(xs: np.ndarray, ys: np.ndarray, length: float, width: float, cell: float = 1.0,
                 sigma_cells: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed 2D histogram normalised to 0..1. Returns (z, x_centres, y_centres)."""
    nx, ny = int(np.ceil(length / cell)), int(np.ceil(width / cell))
    hist, xe, ye = np.histogram2d(xs, ys, bins=[nx, ny], range=[[0, length], [0, width]])
    z = cv2.GaussianBlur(hist.T.astype(np.float32), (0, 0), sigma_cells)
    if z.max() > 0:
        z = z / z.max()
    return z, (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2


def pitch_heatmap(xs, ys, length: float = config.PITCH_LENGTH_M, width: float = config.PITCH_WIDTH_M,
                  title: str | None = None, height: int = 460) -> go.Figure:
    fig = pitch_figure(length, width, title, height)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs):
        z, xc, yc = density_grid(xs, ys, length, width)
        fig.add_trace(go.Heatmap(
            z=z, x=xc, y=yc, colorscale=DENSITY_COLORSCALE, zmin=0, zmax=1, zsmooth="best",
            colorbar=dict(title=dict(text="Relative<br>time spent", font=dict(size=11)), thickness=10,
                          tickvals=[0, 0.5, 1], ticktext=["low", "", "high"]),
            hovertemplate="x %{x:.0f} m, y %{y:.0f} m<br>relative density %{z:.2f}<extra></extra>",
        ))
        # Draw the markings again on top of the density.
        for line in pitch_line_polylines(length, width):
            fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode="lines", hoverinfo="skip",
                                     showlegend=False, line=dict(color=PITCH_LINE, width=1)))
    return fig


def pitch_paths(paths: dict[str, pd.DataFrame], colours: dict[str, str], length: float = config.PITCH_LENGTH_M,
                width: float = config.PITCH_WIDTH_M, title: str | None = None, height: int = 460) -> go.Figure:
    """`paths` maps a label to a DataFrame with sx, sy, time_s columns."""
    fig = pitch_figure(length, width, title, height)
    for label, df in paths.items():
        _add_path_traces(fig, label, df, colours[label])
    return fig


def _add_path_traces(fig: go.Figure, label: str, df: pd.DataFrame, colour: str) -> None:
    if df.empty:
        return
    fig.add_trace(go.Scatter(
        x=df["sx"], y=df["sy"], mode="lines", name=label, line=dict(color=colour, width=2),
        customdata=df["time_s"],
        hovertemplate=f"{label}<br>t %{{customdata:.1f}} s<extra></extra>",
    ))
    end = df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[end["sx"]], y=[end["sy"]], mode="markers+text", text=[label], textposition="top center",
        textfont=dict(color=INK, size=11), marker=dict(size=9, color=colour, line=dict(color=SURFACE, width=2)),
        showlegend=False, hoverinfo="skip",
    ))


# ---------------------------------------------------------------------------
# Image-space (camera view) figures
# ---------------------------------------------------------------------------
def _muted_background(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = (gray.astype(np.float32) * 0.55 + 255 * 0.35).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _ramp_lut() -> np.ndarray:
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def image_heatmap(image_rgb: np.ndarray, xs, ys, sigma_px: float = 18.0) -> tuple[np.ndarray, float]:
    """Density of positions drawn over a muted copy of the reference frame.

    Returns (rgb_image, share_of_points_outside_the_frame).
    """
    h, w = image_rgb.shape[:2]
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    out = _muted_background(image_rgb)
    if len(xs) == 0:
        return out, 0.0
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    outside_share = float(1.0 - inside.mean())
    factor = 4
    grid = np.zeros((h // factor + 1, w // factor + 1), dtype=np.float32)
    np.add.at(grid, ((ys[inside] / factor).astype(int), (xs[inside] / factor).astype(int)), 1.0)
    grid = cv2.GaussianBlur(grid, (0, 0), sigma_px / factor)
    if grid.max() > 0:
        grid /= grid.max()
    density = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    colours = _ramp_lut()[(density * 255).astype(np.uint8)]
    alpha = np.clip(density * 1.6, 0, 0.85)[..., None]
    alpha[density < 0.03] = 0
    blended = (out * (1 - alpha) + colours * alpha).astype(np.uint8)
    return blended, outside_share


def image_paths(image_rgb: np.ndarray, paths: dict[str, pd.DataFrame], colours: dict[str, str],
                title: str | None = None, height: int = 460) -> go.Figure:
    fig = go.Figure(go.Image(z=_muted_background(image_rgb), hoverinfo="skip"))
    for label, df in paths.items():
        _add_path_traces(fig, label, df, colours[label])
    base_layout(fig, title, height)
    _compact_margins(fig, title)
    h, w = image_rgb.shape[:2]
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x")
    return fig


def image_figure(image_rgb: np.ndarray, title: str | None = None, height: int = 520) -> go.Figure:
    """Show an image with pixel coordinates on hover (used for calibration)."""
    fig = go.Figure(go.Image(z=image_rgb, hovertemplate="x %{x}, y %{y}<extra></extra>"))
    base_layout(fig, title, height)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------------------
# Matplotlib exports for the PDF report
# ---------------------------------------------------------------------------
def _mpl_axes(figsize=(7.5, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def save_pitch_heatmap_png(xs, ys, length: float, width: float, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    ax.set_facecolor(PITCH_SURFACE)
    if len(xs):
        z, _, _ = density_grid(np.asarray(xs), np.asarray(ys), length, width)
        cmap = LinearSegmentedColormap.from_list("ramp", ["#ffffff00"] + BLUE_RAMP)
        ax.imshow(np.ma.masked_less(z, 0.03), extent=[0, length, width, 0], cmap=cmap, vmin=0, vmax=1,
                  interpolation="bilinear", zorder=1)
    for line in pitch_line_polylines(length, width):
        ax.plot(line[:, 0], line[:, 1], color=PITCH_LINE, linewidth=1, zorder=2)
    ax.set_xlim(-2, length + 2)
    ax.set_ylim(width + 2, -2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def save_image_png(image_rgb: np.ndarray, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * image_rgb.shape[0] / image_rgb.shape[1] + 0.4), dpi=150)
    ax.imshow(image_rgb)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def save_bar_png(labels, values, path: Path, title: str, ylabel: str) -> Path:
    fig, ax = _mpl_axes()
    ax.bar([str(v) for v in labels], values, color=SERIES[0], width=0.6)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
    if len(labels) > 20:
        ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def save_line_png(x, y, path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = _mpl_axes()
    ax.plot(x, y, color=SERIES[0], linewidth=1.5)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
