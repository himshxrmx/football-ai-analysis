"""CSV and PDF export.

The PDF is built with fpdf2 using the built-in Helvetica font, which only
supports Latin-1 characters, so all text passes through `_latin1()`.
Charts are rendered with Matplotlib into a temporary folder and embedded.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF, XPos, YPos

import config
from modules import heatmap_generator as hg
from modules.coaching_insights import Insight
from modules.run_store import RunData

_REPLACEMENTS = {
    "—": "-", "–": "-", "−": "-", "≥": ">=", "≤": "<=", "’": "'", "‘": "'",
    "“": '"', "”": '"', "…": "...", " ": " ",
}


def _latin1(text) -> str:
    text = str(text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _num(value, digits: int = 1, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{(' ' + unit) if unit else ''}"
    return f"{float(value):,.{digits}f}{(' ' + unit) if unit else ''}"


class _Report(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 16, 16)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, _latin1(self.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, f"Page {self.page_no()}", align="R")

    def h1(self, text: str):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 9, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h2(self, text: str):
        if self.get_y() > self.h - 50:
            self.add_page()
        self.ln(3)
        self.set_font("helvetica", "B", 13)
        self.set_text_color(20, 60, 35)
        self.multi_cell(0, 7, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 205, 195)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def h3(self, text: str):
        if self.get_y() > self.h - 40:  # keep a heading together with its first lines
            self.add_page()
        self.set_font("helvetica", "B", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def para(self, text: str, size: float = 9.5, colour=(40, 40, 40)):
        self.set_font("helvetica", "", size)
        self.set_text_color(*colour)
        self.multi_cell(0, 5, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def key_values(self, rows: list[tuple[str, str]]):
        self.set_font("helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        with self.table(col_widths=(55, 120), text_align="LEFT", first_row_as_headings=False,
                        line_height=5.5, borders_layout="HORIZONTAL_LINES") as table:
            for key, value in rows:
                row = table.row()
                row.cell(_latin1(key))
                row.cell(_latin1(value))
        self.ln(2)

    def data_table(self, header: list[str], rows: list[list[str]], widths: tuple):
        self.set_font("helvetica", "", 8)
        self.set_text_color(40, 40, 40)
        with self.table(col_widths=widths, text_align="LEFT", line_height=5, first_row_as_headings=True,
                        borders_layout="HORIZONTAL_LINES") as table:
            row = table.row()
            for h in header:
                row.cell(_latin1(h))
            for values in rows:
                row = table.row()
                for v in values:
                    row.cell(_latin1(v))
        self.ln(2)

    def image_full(self, path: Path, max_height: float = 110):
        from PIL import Image

        with Image.open(path) as im:
            w_px, h_px = im.size
        width = self.epw
        height = width * h_px / w_px
        if height > max_height:
            height = max_height
            width = height * w_px / h_px
        if self.get_y() + height > self.h - 18:
            self.add_page()
        self.image(str(path), x=self.l_margin + (self.epw - width) / 2, w=width, h=height)
        self.ln(3)


def _insight_block(pdf: _Report, insights: list[Insight]) -> None:
    labels = {"observation": "Observations (measured)", "suggestion": "Suggestions (hypotheses, may be wrong)",
              "data_quality": "Data quality notes"}
    for kind in ("data_quality", "observation", "suggestion"):
        items = [i for i in insights if i.kind == kind]
        if not items:
            continue
        pdf.h3(labels[kind])
        for ins in items:
            pdf.para(f"{ins.title}: {ins.detail}", 9)
            why = f"Rule {ins.rule_id}: {ins.rule}"
            if ins.drill:
                why += f" Drill: {ins.drill['name']} ({ins.drill['purpose']})"
            if ins.caveat:
                why += f" Caveat: {ins.caveat}"
            pdf.para(why, 8, (100, 100, 100))
            pdf.ln(1)


def build_pdf_report(run: RunData, metrics: pd.DataFrame, overview: dict, steps: pd.DataFrame,
                     team: list[Insight], players: dict[int, list[Insight]], out_path: Path) -> Path:
    """Create the PDF report and return its path."""
    meta = run.meta
    video = meta["video"]
    proc = meta["processing"]
    settings = meta["settings"]
    space = "pitch" if run.is_calibrated else "image"
    unit = "m" if run.is_calibrated else "px"
    speed_unit = "m/s" if run.is_calibrated else "px/s"

    pdf = _Report(f"Football video analysis: {video['file_name']}")
    pdf.add_page()
    pdf.h1("Football Video Analysis Report")
    pdf.para(f"Video: {video['file_name']}    Analysis created: {meta['created']}    Report created: "
             f"{time.strftime('%Y-%m-%d %H:%M')}", 9, (90, 90, 90))
    pdf.ln(2)
    pdf.para(
        "This report summarises what the system measured from the video: detected and tracked people, "
        "their positions and the movement derived from them. Tracking IDs are arbitrary numbers and do not "
        "identify real players. Passes, shots, goals and possession are not measured."
    )

    pdf.h2("Video and processing")
    pdf.key_values([
        ("Resolution / frame rate", f"{video['width']} x {video['height']} px, {video['fps']:.2f} fps"),
        ("Video length", f"{video['duration_s']:.1f} s ({video['frame_count']} frames)"),
        ("Processed", f"{proc['duration_processed_s']:.1f} s, every {settings['frame_stride']} frame(s), "
                      f"{proc['frames_processed']} frames"),
        ("Model", f"{meta['model']['name']} ({meta['model']['type']} classes), input size {settings['image_size']} px, "
                  f"confidence threshold {settings['confidence']:.2f}"),
        ("Tracker", "ByteTrack (ultralytics implementation)"),
        ("Camera motion compensation", f"on, successful steps {proc['camera_success_rate'] * 100:.0f}%"
         if proc["camera_compensation"] else "off"),
        ("Pitch calibration", f"yes, mean reprojection error {run.calibration.reprojection_error_m:.2f} m"
         if run.is_calibrated else "no (movement values are in image pixels)"),
        ("Processing time", f"{proc['elapsed_s']:.0f} s ({proc['processing_fps']} frames per second)"),
    ])

    pdf.h2("Summary")
    pdf.key_values([
        ("Track IDs created", str(overview["unique_track_ids"])),
        (f"Stable tracks (>= {config.MIN_TRACK_SECONDS:.0f} s)", str(overview["stable_tracks"])),
        ("People tracked per frame", f"mean {overview['mean_tracked_per_frame']}, max {overview['max_tracked_in_frame']}"),
        ("Frames without tracked people", f"{overview['frames_without_players']} ({overview['frames_without_players_pct']}%)"),
        ("Mean detection confidence", _num(overview["mean_confidence"], 2)),
        ("Frames with a ball detection", f"{overview['ball_detection_pct']}%"),
        ("Scene cuts detected", str(overview["scene_cuts"])),
    ])
    pdf.para("The number of track IDs is normally higher than the number of people on the pitch, because a "
             "person who is hidden for longer than the track buffer receives a new ID.", 8.5, (90, 90, 90))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames = run.frames
        if not frames.empty:
            pdf.h2("Tracked people per frame")
            pdf.image_full(hg.save_line_png(frames["time_s"], frames["tracked_players"], tmp / "per_frame.png",
                                            "Tracked people per processed frame", "time (s)", "people"), 70)
        stable = metrics[metrics["stable"]].sort_values("distance", ascending=False)
        if not stable.empty:
            pdf.h2(f"Distance while tracked ({'metres, calibrated estimate' if run.is_calibrated else 'image pixels'})")
            top = stable.head(25)
            pdf.image_full(hg.save_bar_png([f"ID {i}" for i in top["track_id"]], top["distance"], tmp / "distance.png",
                                           "Distance by track ID (stable tracks, top 25)", unit), 75)
        if not steps.empty:
            pdf.h2("Overall activity heatmap")
            if run.is_calibrated:
                cal = run.calibration
                path = hg.save_pitch_heatmap_png(steps["sx"], steps["sy"], cal.pitch_length, cal.pitch_width,
                                                 tmp / "heatmap.png", "All tracked positions, top-down pitch view")
            else:
                ref = cv2.cvtColor(cv2.imread(str(run.reference_frame)), cv2.COLOR_BGR2RGB)
                img, outside = hg.image_heatmap(ref, steps["sx"], steps["sy"])
                path = hg.save_image_png(img, tmp / "heatmap.png", "All tracked positions over the reference camera view")
                if outside > 0.01:
                    pdf.para(f"{outside * 100:.0f}% of positions fall outside the reference view and are not drawn.", 8.5)
            pdf.image_full(path, 105)

    pdf.h2("Player statistics (stable tracks)")
    pdf.para(f"Values are in {'metres (calibrated estimate)' if run.is_calibrated else 'image pixels (not metres)'}. "
             "Peak speed is the 95th percentile of smoothed speed, which ignores single-frame spikes.", 8.5, (90, 90, 90))
    rows = []
    for r in metrics[metrics["stable"]].sort_values("visible_s", ascending=False).head(40).itertuples():
        rows.append([f"{r.track_id}", r.colour_group, _num(r.visible_s, 1), _num(r.distance, 0),
                     _num(r.avg_speed, 2), _num(r.peak_speed, 2), _num(r.area_covered, 0)])
    if rows:
        pdf.data_table(["ID", "Colour group", "Visible s", f"Distance {unit}", f"Avg {speed_unit}",
                        f"Peak {speed_unit}", f"Area {unit}²"], rows, (12, 28, 22, 28, 28, 28, 32))
    else:
        pdf.para("No track was followed long enough to be included.")

    pdf.h2("Coaching insights")
    pdf.para("Generated by transparent rules from the measured values. Observations are measurements; "
             "suggestions are hypotheses for a coach to consider and can be wrong. Fitness, fatigue, injuries, "
             "emotions and tactical effectiveness are not assessed.", 8.5, (90, 90, 90))
    pdf.h3("Whole clip")
    _insight_block(pdf, team)
    for track_id, insights in players.items():
        pdf.ln(1)
        pdf.h3(f"Track ID {track_id}")
        _insight_block(pdf, insights)

    pdf.h2("Method and limitations")
    for line in [
        "Detection: a pretrained YOLO model finds people (and the ball) in each processed frame.",
        "Tracking: ByteTrack links detections over time with a Kalman filter and IoU matching.",
        "Camera motion: optical flow on the grass estimates how the camera moved, so positions are expressed "
        "in one reference view." if proc["camera_compensation"] else "Camera motion compensation was switched off.",
        "Positions are the bottom centre of each box and are smoothed before distances are summed.",
        "A COCO model cannot tell players, referees and coaches apart; colour groups are clusters of shirt colours.",
        "Occlusion, players leaving the view and scene cuts break tracks and lower the reliability of all values.",
        "Without calibration, distances and speeds are in pixels and must not be read as metres.",
    ]:
        pdf.para(f"- {line}", 9)
    pdf.output(str(out_path))
    return Path(out_path)


def metrics_csv(metrics: pd.DataFrame) -> bytes:
    return metrics.to_csv(index=False).encode("utf-8")
