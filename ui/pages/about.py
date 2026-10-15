"""Page: methodology, measured performance and limitations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from ui import components


def render() -> None:
    components.page_header("About and method", "How the system works, what it measures, what it estimates and what it cannot do.")

    st.header("Processing pipeline")
    st.markdown("""
1. **Validation.** The file type, size, duration and frame rate are checked, and the first and a middle frame are
   decoded to prove the file is readable.
2. **Frame reading.** OpenCV reads every Nth frame (the frame step). Skipped frames are not decoded.
3. **Detection.** A pretrained YOLO model returns boxes and confidence scores for people and the ball.
4. **Pitch filter.** The largest green region of the frame is found; people whose feet are outside it (crowd,
   benches) are removed.
5. **Tracking.** ByteTrack links boxes across frames with a Kalman filter and IoU matching and gives each track an ID.
6. **Camera motion compensation.** Points on the grass are followed with Lucas-Kanade optical flow; RANSAC fits a
   homography per frame, so all positions can be expressed in one reference view.
7. **Recording.** For every tracked box the time, ID, confidence, box, feet position, reference-view position and
   shirt colour are saved to CSV.
8. **Analysis.** Positions are smoothed; distance, speed, area covered, zones and colour groups are computed.
   With a pitch calibration the values are converted to metres.
9. **Presentation.** Dashboards, heatmaps, rule-based coaching insights, CSV and PDF exports.
""")

    st.header("Models and classes")
    st.markdown(f"""
* **Default model:** `{config.DEFAULT_MODEL}` (Ultralytics YOLO26 nano, trained on the COCO dataset). Used classes:
  `person` (class 0) and `sports ball` (class 32). A COCO model cannot tell players, goalkeepers, referees and
  coaches apart.
* **Optional football model:** a model fine-tuned on football classes (player, goalkeeper, referee, ball) can be
  placed at `models/football.pt`. Its class names are read from the weights and mapped automatically.
* **Tracker:** ByteTrack, from the ultralytics package, with an IoU match tolerance tuned for small players.
* **Pose (optional):** `{config.POSE_MODEL}`, 17 COCO body keypoints.
""")

    st.header("Measured on the development laptop")
    st.caption("Intel Core i7-8750H CPU, 8 GB RAM, no GPU. Sample: 30 s Bundesliga clip, 1920 x 1080, 25 fps.")
    st.dataframe(pd.DataFrame([
        ("Detection time per frame, 640 px input", "about 100 ms"),
        ("Detection time per frame, 960 px input", "about 165 ms"),
        ("Detection time per frame, 1280 px input", "about 275 ms"),
        ("People detected per frame (10 sampled frames), 640 px", "16.4 on average"),
        ("People detected per frame (10 sampled frames), 960 px", "21.9 on average"),
        ("Full pipeline with default settings", "2.9 frames per second (30 s clip in about 2 min 10 s)"),
        ("Track IDs on the sample, ByteTrack default matching", "110"),
        ("Track IDs on the sample, tuned matching (used)", "49"),
        ("Track IDs on the sample, BoT-SORT with camera compensation", "55, about 15 times slower"),
    ], columns=["Measurement", "Result"]), hide_index=True, width="stretch")
    st.markdown("""
**Speed versus accuracy.** A larger model input finds small, distant players more reliably (21.9 instead of 16.4
people per frame at 960 px compared with 640 px) but takes longer. Analysing every second frame halves the work;
larger steps are faster but players move further between analysed frames, so tracks break more often.
""")

    st.header("What is measured and what is estimated")
    st.markdown("""
| Output | Status |
|---|---|
| Boxes, confidence scores, tracking IDs, time in view | Measured directly by the model and tracker |
| Positions in the reference camera view | Measured, corrected for camera motion (small drift over long clips) |
| Distance and speed in metres, sprints | Estimated, only with a pitch calibration |
| Distance and speed without calibration | Image pixels; not metres and not comparable between near and far players |
| Colour groups | Experimental clustering of shirt colours, not verified teams |
| Coaching suggestions | Hypotheses from transparent rules |
| Passes, shots, goals, assists, possession | Not measured |
| Fitness, fatigue, injury, emotion, tactical quality | Not assessed |
""")

    st.header("Known limitations")
    st.markdown("""
* Occlusion and players leaving the view create new IDs; the number of IDs is higher than the number of players.
* The ball is small and fast; the COCO model misses it in most frames of wide shots.
* Camera motion compensation drifts slowly and breaks at scene cuts and replays.
* Calibration assumes a flat pitch and accurate landmark clicks; check the projected lines on the calibration page.
* Shirt-colour grouping fails when kits are similar or lighting changes strongly.
* Processing on a CPU is slow for long videos; analyse short clips.
""")

    st.header("Future work")
    st.markdown("""
* Fine-tune YOLO on a labelled football dataset to separate players, goalkeepers, referees and the ball.
* Automatic pitch keypoint detection for per-frame calibration of broadcast footage.
* Re-identification features to keep IDs across long occlusions, and shirt-number recognition.
* Ball tracking with interpolation, followed by validated event detection (passes, shots).
* GPU support for real-time processing.
""")

    st.header("Source code layout")
    st.code("""app.py                    Streamlit entry point and navigation
config.py                 all settings in one place
modules/                  processing (no Streamlit code)
  video_processor.py      validation, reading, H.264 writing
  player_detector.py      YOLO detection, class mapping, pitch filter
  player_tracker.py       ByteTrack wrapper, movement trails
  camera_motion.py        optical-flow homography, scene-cut check
  pipeline.py             end-to-end analysis and saved outputs
  movement_analyzer.py    distances, speeds, zones, summaries
  team_classifier.py      shirt-colour grouping
  pitch_calibration.py    landmarks and homography to metres
  heatmap_generator.py    heatmaps, paths, pitch drawings, PDF charts
  coaching_insights.py    rule-based insights and drills
  report_generator.py     PDF report
  pose_estimator.py       optional pose estimation
ui/                       Streamlit pages and components
tests/                    automated tests (pytest)""", language="text")
