# Football Video Analysis System

A web application that analyses football match videos with computer vision. It detects people with a pretrained
YOLO model, tracks them with ByteTrack, compensates for camera movement, converts positions to metres after a
pitch calibration, and presents heatmaps, movement statistics, player comparisons, rule-based coaching insights
and CSV/PDF exports in a Streamlit dashboard.

Computer Engineering (AI & ML) college project. Runs on a normal laptop CPU; no GPU and no paid API are needed.

---

## Run it on your own computer

Everything runs locally: your videos are processed on your own computer and never uploaded anywhere.

**You need**

* Windows 10/11, macOS on Apple Silicon (M1 or newer), or Linux, with 8 GB RAM and about 3 GB of free disk space.
  No graphics card is needed. (Current PyTorch versions no longer support Intel-based Macs.)
* **Python 3.11, 3.12 or 3.13** (3.13 recommended). On Windows, install it from
  [python.org/downloads](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** in the installer.
* An internet connection for the first setup (about 1 GB of packages, plus a 5 MB model file on first use).

**Step 1: Get the files**

* Easiest: on this repository's GitHub page click the green **Code** button, then **Download ZIP**, and extract
  the ZIP to a folder such as your Desktop.
* Or, with Git installed:

```bash
git clone https://github.com/himshxrmx/football-ai-analysis.git
```

**Step 2: Install (only once, takes about 5 to 15 minutes)**

* **Windows:** open the extracted folder and double-click **`setup.bat`**. If Windows shows "Windows protected your
  PC", click **More info**, then **Run anyway** (the file only creates a `.venv` folder and installs the packages
  listed in `requirements.txt`).
* **macOS / Linux:** open a terminal in the folder and run:

```bash
chmod +x setup.sh run_app.sh
```

```bash
./setup.sh
```

The setup creates a private Python environment in the `.venv` folder, installs the CPU version of PyTorch and the
other packages, and downloads a 30-second sample football clip.

**Step 3: Start the app**

* **Windows:** double-click **`run_app.bat`**.
* **macOS / Linux:** run `./run_app.sh` in the terminal.

Your browser opens **http://localhost:8501** after a few seconds (open it yourself if it does not). Keep the black
command window open while you use the app; closing it stops the app.

**Step 4: Try it**

1. On **Analyse a video**, choose **Use a sample video** (or upload your own MP4, AVI, MOV or MKV clip).
2. Click **Run detection on this frame** to test the model, then **Run analysis**. The 30-second sample takes about
   2 minutes on a laptop CPU. The first run downloads the 5 MB YOLO model automatically.
3. Explore **Overview**, **Player tracking**, **Heatmaps and movement**, **Performance statistics**,
   **Player comparison** and **AI coach**.
4. For values in metres, open **Pitch calibration** and enter at least four pitch landmarks. The values for the
   sample clip are listed in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).
5. Download the PDF report and CSV files on **Reports and export**.

Results are saved in the `outputs` folder; uploaded videos in `data/uploads`. Delete those folders' contents to
remove them.

### Troubleshooting

| Problem | Solution |
|---|---|
| "Python 3.11, 3.12 or 3.13 was not found" | Install Python 3.13 from python.org (tick "Add python.exe to PATH"), close and reopen the folder, run setup again. |
| Setup stops with a red pip error | Check the internet connection and free disk space, then run setup again; it continues where it stopped. |
| "The virtual environment is missing" | Run `setup.bat` (Windows) or `./setup.sh` (macOS / Linux) first. |
| Browser shows "This site can't be reached" | Wait until the command window shows "You can now view your Streamlit app", then reload the page. |
| Port 8501 is already in use | Close the other app window, or start on another port: `run_app.bat --server.port 8502` and open http://localhost:8502. |
| Model download fails on the first analysis | Allow internet access once, or download [yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) manually into the `models` folder. |
| Sample video download fails | Upload your own short football clip on the Analyse page instead. |
| Linux: `libGL.so.1: cannot open shared object file` | Install the system libraries: `sudo apt install libgl1 libglib2.0-0`. |
| Analysis is slow | Lower the processing resolution to 640 px, analyse every 3rd frame, or analyse fewer seconds. |

---

## Contents

1. [Abstract](#1-abstract)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Limitations of existing systems](#4-limitations-of-existing-systems)
5. [Proposed system](#5-proposed-system)
6. [Hardware and software requirements](#6-hardware-and-software-requirements)
7. [System architecture](#7-system-architecture)
8. [Data flow and processing pipeline](#8-data-flow-and-processing-pipeline)
9. [Technologies used](#9-technologies-used)
10. [Installation](#10-installation)
11. [Running the application](#11-running-the-application)
12. [Features](#12-features)
13. [Testing](#13-testing)
14. [Results on the sample clip](#14-results-on-the-sample-clip)
15. [Known limitations](#15-known-limitations)
16. [Future enhancements](#16-future-enhancements)
17. [Project structure](#17-project-structure)
18. [Further documents](#18-further-documents)
19. [Licence](#licence)

---

## 1. Abstract

Professional clubs measure player movement with expensive multi-camera or GPS systems. This project shows how
far a single ordinary video and free, open-source AI models can go. Each analysed frame passes through a YOLO26
object detector (COCO-pretrained) that finds people and the ball. ByteTrack links the detections over time into
tracks with IDs. Lucas-Kanade optical flow on the grass estimates how the broadcast camera pans and zooms, so all
positions can be expressed in one reference view, and a homography from manually marked pitch landmarks converts
them to metres. From these positions the system computes distance, speed, sprints, area covered and zone
occupancy, draws heatmaps and movement paths, groups players by shirt colour, and produces explainable,
rule-based coaching observations and suggestions. Everything is exported as CSV and PDF. On a 30-second 1080p
Bundesliga clip the system tracked on average 22.8 people per frame at 3.1 frames per second on a laptop CPU.

## 2. Problem statement

Amateur teams, schools and students have video of their matches but no affordable way to turn it into
objective movement data. Manual analysis is slow and subjective. Commercial tracking products need special
cameras or wearable sensors. The problem is to build a system that takes an ordinary football video and
automatically produces trustworthy, clearly labelled movement information, while being honest about what a
single camera and a general-purpose model can and cannot measure.

## 3. Objectives

1. Accept and validate football videos (type, size, duration, readability) and preview them in the browser.
2. Detect players with a pretrained deep-learning detector and show boxes and confidence scores.
3. Track players across frames with consistent IDs and draw movement trails.
4. Save timestamped tracking data as CSV.
5. Produce individual and overall heatmaps and a top-down pitch view.
6. Estimate physical distance and speed only when a camera-to-pitch calibration exists.
7. Present the results in an interactive dashboard with player comparison.
8. Generate explainable coaching insights that separate measured facts from suggestions.
9. Export results to CSV and PDF.
10. Run on an ordinary laptop without a GPU or paid services.

## 4. Limitations of existing systems

| Approach | Limitation |
|---|---|
| Manual video analysis | Slow, subjective, does not scale, no numeric movement data |
| GPS / IMU wearables | Hardware per player, cost, not available for opponents or archive footage |
| Optical tracking products (multi-camera stadium systems) | Very expensive, fixed installations, closed data |
| Generic cloud vision APIs | Paid per frame, not football-specific, no tracking or pitch geometry, data leaves the device |
| Many student demos | Show random or simulated statistics, ignore camera motion, report pixels as metres |

## 5. Proposed system

* A modular Python pipeline (detection, tracking, camera compensation, calibration, analysis) with no UI code,
  so every part can be tested on its own.
* A Streamlit dashboard with pages for analysis, overview, tracking, heatmaps, statistics, comparison, coaching,
  reports, calibration, pose and project information.
* Transparent units: values are in pixels until the pitch is calibrated, and every page says which.
* Rule-based coaching: every insight states its rule and the evidence values; suggestions carry caveats and are
  only produced when the data is good enough.
* Fully local: videos never leave the computer.

## 6. Hardware and software requirements

**Hardware (minimum, tested):**

* CPU: any 64-bit quad-core (tested on Intel Core i7-8750H, 6 cores)
* RAM: 8 GB
* Disk: about 3 GB for the environment and models, plus space for videos and results
* GPU: not required (an NVIDIA GPU makes processing much faster if a CUDA build of PyTorch is installed)

**Software:**

* Windows 10/11, macOS or Linux
* Python 3.11 to 3.13 (tested with 3.13.7)
* A modern browser (tested with Microsoft Edge 155)
* Internet connection once, to install packages and download model weights
* Google Colab (optional) for training a football-specific model

## 7. System architecture

```
                 +--------------------------------------------------------------+
                 |                  Streamlit user interface (ui/)              |
                 |  Analyse | Overview | Tracking | Heatmaps | Statistics |     |
                 |  Comparison | AI coach | Reports | Calibration | Pose | Info |
                 +-------------------------------+------------------------------+
                                                 | calls
+------------------------------------------------v-------------------------------------------+
|                               Processing modules (modules/)                               |
|                                                                                            |
|  video_processor --> player_detector --> player_tracker --> camera_motion --> pipeline    |
|  (validate, read,    (YOLO26, class      (ByteTrack,        (optical flow,    (orchestrates|
|   write H.264)        map, pitch filter)  trails)            homography)       and saves)  |
|                                                                                            |
|  run_store --> movement_analyzer --> heatmap_generator     coaching_insights               |
|  (load runs)   (metres/pixels,       (Plotly, Matplotlib)  (rules, drills)                 |
|                speed, zones)         team_classifier       report_generator (PDF)          |
|                pitch_calibration     (shirt colours)       pose_estimator (optional)       |
+------------------------------------------------+-------------------------------------------+
                                                 | reads / writes
                 +-------------------------------v------------------------------+
                 |  Local files: data/uploads, data/samples, models, outputs/   |
                 |  outputs/<run>/ meta.json tracks.csv frames.csv ball.csv     |
                 |                 annotated.mp4 reference_frame.jpg            |
                 |                 calibration.json report.pdf                  |
                 +--------------------------------------------------------------+
```

Storage uses local files instead of a database: every analysis is a self-contained folder that can be opened,
copied or deleted, and CSV files open directly in Excel or pandas.

## 8. Data flow and processing pipeline

1. **Upload / select** a video (MP4, AVI, MOV or MKV).
2. **Validate**: extension, size (500 MB limit), duration (1 s to 10 min), frame rate, and decoding of the first
   and a middle frame.
3. **Read** every Nth frame with OpenCV (skipped frames are only grabbed, not decoded).
4. **Detect** people and the ball with YOLO26 at the chosen input size. Low-confidence boxes (down to 0.10) are
   kept for the tracker's second association step.
5. **Pitch filter**: the largest green region's convex hull is the playing area; people whose feet are outside it
   (crowd, bench) are dropped.
6. **Track** with ByteTrack: Kalman prediction, IoU matching of high-confidence boxes, then of low-confidence
   boxes, new tracks from unmatched confident boxes, removal of tracks lost for more than 2 s.
7. **Camera motion**: Shi-Tomasi corners on the grass (player boxes masked out) are followed into the next frame
   with Lucas-Kanade optical flow; RANSAC fits a homography; homographies are chained to the first frame and later
   re-expressed in the widest (reference) frame.
8. **Record** for every tracked box: time, ID, box, confidence, feet point, reference-view position and median
   shirt colour (CIE Lab, grass pixels removed).
9. **Annotate** the frame (boxes, IDs, confidence, trails projected through the camera homography, ball marker)
   and write an H.264 MP4 at up to 1280 px width.
10. **Save** tracks.csv, frames.csv, ball.csv, meta.json and reference_frame.jpg.
11. **Calibrate** (optional): at least four pitch landmarks give a homography from reference pixels to metres.
12. **Analyse**: smoothing (5-frame moving average that preserves straight-line motion), step distances (gaps
    over 3 frames and steps over 12 m/s rejected), speed, 95th-percentile peak speed, high-speed distance
    (> 5.5 m/s), sprints (> 7 m/s for at least 1 s), convex-hull area, zones, colour groups.
13. **Present** in the dashboard, **coach** with rules, **export** CSV and PDF.

**Speed versus accuracy.** The frame step and the model input size are the two main controls:

| Setting | Faster | More accurate |
|---|---|---|
| Model input size | 640 px (about 100 ms per frame) | 960 px (about 165 ms) or 1280 px (about 275 ms) |
| Frame step | every 3rd to 5th frame | every frame or every 2nd frame |
| Model | yolo26n (nano) | yolo26s (small) |

At 640 px the model found 16.4 people per frame on sampled frames of the demo clip, at 960 px 21.9, because
distant players are only 20 to 40 pixels tall. Larger frame steps make the movement between analysed frames
larger, which lowers box overlap and causes more ID changes.

## 9. Technologies used

| Technology | Use |
|---|---|
| Python 3.13 | Language |
| OpenCV 5 | Video I/O, optical flow, homographies, colour spaces, k-means |
| Ultralytics YOLO26 (8.4.177) | Pretrained detector and pose model, ByteTrack implementation |
| PyTorch 2.14 (CPU) | Neural network runtime |
| NumPy, pandas | Numerical work and tabular data |
| Plotly | Interactive charts in the dashboard |
| Matplotlib | Static charts for the PDF |
| Streamlit 1.66 | Web interface |
| fpdf2 | PDF reports |
| imageio-ffmpeg | Bundled FFmpeg for browser-playable H.264 video |
| pytest | Automated tests |
| Google Colab (optional) | Training a football-specific detector |

**Model and classes.** The default model is `yolo26n.pt`, trained on COCO (80 classes). The project uses class 0
`person` and class 32 `sports ball`. A COCO model does not distinguish players, goalkeepers, referees or coaches.
A model fine-tuned on football classes (`player`, `goalkeeper`, `referee`, `ball`) can be placed at
`models/football.pt`; its class names are read from the weights and mapped automatically (see
`notebooks/train_football_yolo_colab.ipynb`).

**Model weights.** Weights are not stored in the repository. The official files (`yolo26n.pt` 5.3 MB,
`yolo26n-pose.pt` 7.5 MB, optional `yolo26s.pt`, `yolo11n.pt`) are downloaded automatically by the ultralytics
package from the Ultralytics GitHub release page into `models/` the first time they are used.

## 10. Installation

The quickest way is the setup script described in [Run it on your own computer](#run-it-on-your-own-computer)
(`setup.bat` on Windows, `setup.sh` on macOS and Linux). The same steps by hand, in a terminal opened in the
project folder:

**Windows:**

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
```

```bash
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe scripts\download_sample.py
```

**macOS / Linux** (on Linux, first install the CPU build of PyTorch with the same `--index-url` command as above,
using `.venv/bin/python`):

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```bash
.venv/bin/python scripts/download_sample.py
```

## 11. Running the application

Windows: double-click `run_app.bat`. macOS / Linux: `./run_app.sh`. Or start Streamlit directly:

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501 in the browser. The first analysis downloads the model weights (about 5 MB).

**Typical demonstration flow**

1. Analyse a video: choose "Use a sample video", check the preview, run "Run detection on this frame", then
   "Run analysis" (about 2 minutes for the 30 s sample on a laptop CPU).
2. Pitch calibration: enter at least four landmarks (the sample's values are listed in `docs/DEMO_SCRIPT.md`)
   and save. All pages switch to metres.
3. Overview, Player tracking, Heatmaps, Performance statistics, Player comparison, AI coach.
4. Reports and export: build the PDF and download the CSV files.

**Optional settings (environment variables)**

| Variable | Effect |
|---|---|
| `FOOTBALL_MODEL_PATH` | Path to a custom football model |
| `FOOTBALL_DEVICE` | `cpu` (default) or `0` for the first CUDA GPU |
| `COACH_LLM_ENABLED=1` | Shows the optional written summary using a local Ollama model |
| `OLLAMA_URL`, `OLLAMA_MODEL` | Ollama server and model (default `http://localhost:11434`, `llama3.2`) |
| `FOOTBALL_CONTACT_EMAIL` | Contact address shown on the privacy and terms pages |

## 12. Features

**A. Video upload and processing.** File type, size, duration and readability checks with clear error messages;
in-browser preview (non-H.264 files are transcoded for the preview); configurable confidence, input size, frame
step, duration, ball detection, pitch filter, camera compensation and trail length; progress bar with frames
per second and time left; download of the annotated video.

**B. Player detection.** YOLO26 boxes with confidence scores; single-frame detection check with timing;
documented class mapping; frames without detections are handled normally.

**C. Player tracking.** ByteTrack IDs, movement trails that stay on the ground while the camera moves, unique ID
count with an explanation of why it exceeds the number of players, timestamped CSV.

**D. Movement analysis and heatmaps.** Overall and individual heatmaps, movement paths, top-down pitch view after
calibration, camera-view heatmaps before calibration. Distances in metres only with calibration; otherwise
clearly labelled pixels.

**E. Dashboard.** Overview, Player tracking, Heatmaps, Performance statistics (sortable table, filters, charts,
speed over time, zones), Player comparison (2 to 4 IDs, small multiples, speed and paths), AI coach, Reports.

**F. AI coaching insights.** Observations (O1 to O5, T1 to T3), suggestions (S1 to S7) with drills and caveats,
data-quality notes (Q0 to Q6). Suggestions need a calibration, at least 5 s or 20% of the clip of tracking, and
at least 3 other stable tracks to compare with; zone suggestions need at least 60 s. Nothing about fitness,
fatigue, injury, emotion, tactics, passes, shots, goals or possession is inferred. An optional local language
model (Ollama) can write a summary of the computed facts.

**G. Exports.** PDF report; CSV for tracks, frames, ball, player statistics, zones and insights; annotated MP4;
heatmap PNG; every chart can be saved as PNG from its toolbar.

**H. Optional extensions.** Pose estimation for one player in one frame (YOLO26 pose, 17 keypoints, 2D joint
angles); experimental shirt-colour groups (k-means in CIE Lab).

**Interface rules.** No decorative gradients, emoji or invented numbers; every value on screen comes from the
analysis. A favicon, a privacy policy page and a terms of use page are included, and the Streamlit developer
toolbar and usage statistics are switched off.

## 13. Testing

Run all automated tests (56 tests, about 25 seconds):

```bash
.venv\Scripts\python.exe -m pytest
```

| Test file | What it checks |
|---|---|
| `test_video_processor.py` | Unsupported type, empty, missing, corrupted, too large and too short files; video information; frame reading with a step; H.264 writing; safe file names |
| `test_detection_tracking.py` | COCO and football class mapping; missing and corrupted model files raise clear errors; detection on a reference image; empty frames; pitch filter; ByteTrack ID consistency, two players, empty frames, ID reset; trails |
| `test_analysis.py` | Camera translation recovery; scene-cut measure; calibration accuracy and rejection of bad input; distances and speeds against known motion; sprint detection; rejected impossible speeds; gaps; empty data; zones; colour groups; coaching rules and their conditions; PDF generation with and without data |
| `test_pipeline.py` | Frame planning; invalid video; complete pipeline on a synthetic video with no players |

Tests that need the model weights are skipped automatically if the weights have not been downloaded yet.

**Manual test with a short sample video**

1. Run `scripts/download_sample.py`, start the app, choose the sample.
2. Check that the video information shows 1920 x 1080, 25 fps, 30.0 s, 750 frames.
3. Run the detection check on frame 125: about 20 to 25 people and a detection time should appear.
4. Run the analysis with the default settings; the overview should show 0 frames without tracked people.
5. Upload a renamed text file as `.mp4`: an error message must appear and the app must keep working.

## 14. Results on the sample clip

30 s Bundesliga clip (1920 x 1080, 25 fps), default settings (YOLO26 nano, 960 px, every 2nd frame), laptop CPU:

| Measure | Value |
|---|---|
| Frames analysed | 375 |
| Processing time | 120 s (3.1 frames per second) |
| People tracked per frame | mean 22.8, max 26 |
| Track IDs created / stable (at least 2 s) | 49 / 37 |
| Camera motion estimated | 100% of frame steps |
| Frames with a ball detection | 34.1% |
| Calibration (7 landmarks) | mean reprojection error 0.22 m |
| Average speed of stable tracks | median 2.58 m/s |
| Highest peak speed (95th percentile) | 7.23 m/s |

**Tracker tuning experiment** (same detections, 30 s clip):

| Tracker setting | Track IDs |
|---|---|
| ByteTrack, default match threshold 0.8 | 110 |
| ByteTrack, match threshold 0.9 (used) | 49 |
| BoT-SORT with sparse optical flow camera compensation | 55 (about 15 times slower) |

## 15. Known limitations

* IDs change after occlusion or when a player leaves the view; the number of IDs is higher than the number of
  players, and IDs do not identify players.
* A COCO model cannot separate players, referees and staff; colour groups are an approximation.
* The ball is small; it is detected in about a third of the frames of a wide shot.
* Camera compensation drifts slowly and breaks at scene cuts and replays; a broadcast highlight with many cuts is
  not suitable.
* Calibration is manual, assumes a flat pitch and is only as good as the clicked points.
* Pixel-based values (without calibration) are distorted by perspective.
* CPU processing is slow for long videos (about 4 minutes per minute of video at the default settings).
* Pose estimation works only for players that are large enough in the frame.

## 16. Future enhancements

* Fine-tune YOLO on a labelled football dataset (players, goalkeepers, referees, ball) in Google Colab.
* Automatic pitch keypoint detection for per-frame calibration of broadcast video.
* Re-identification and shirt-number recognition to keep identities across occlusions.
* Ball tracking with interpolation, then validated event detection (passes, shots).
* Team formation and pressing analysis built on calibrated positions.
* GPU and video-stream support for near real-time analysis.
* Deployment with user accounts on a server (then the privacy policy must be updated).

## 17. Project structure

```
football-ai-analysis/
  app.py                     Streamlit entry point and navigation
  config.py                  all settings in one place
  LICENSE                    GNU AGPL-3.0
  requirements.txt           pinned dependencies
  setup.bat, setup.sh        one-time installation (Windows / macOS and Linux)
  run_app.bat, run_app.sh    start the app (Windows / macOS and Linux)
  pytest.ini                 test configuration
  .streamlit/config.toml     theme, upload limit, toolbar and statistics settings
  assets/                    favicon and logo
  modules/
    video_processor.py       validation, reading, H.264 writing, previews
    player_detector.py       YOLO loading, class mapping, detection, pitch filter
    player_tracker.py        ByteTrack wrapper and movement trails
    camera_motion.py         optical-flow homographies, scene-cut measure
    pipeline.py              end-to-end analysis and saved outputs
    run_store.py             listing and loading saved analyses
    pitch_calibration.py     landmarks, homography to metres, overlay
    movement_analyzer.py     smoothing, distance, speed, sprints, zones, overview
    team_classifier.py       shirt colour measurement and k-means groups
    heatmap_generator.py     heatmaps, paths, pitch drawing, PDF charts
    coaching_insights.py     rules, drills, optional local language model
    report_generator.py      PDF report
    pose_estimator.py        optional pose estimation
  ui/
    theme.py, state.py, components.py
    pages/                   one module per page
  tests/                     pytest suite
  scripts/download_sample.py
  notebooks/train_football_yolo_colab.ipynb
  docs/                      viva questions, presentation, demo script, algorithms
  data/samples/, data/uploads/, models/, outputs/   (contents not committed)
```

## Project team

The privacy policy and terms pages refer contact questions to the people listed here. Fill in before submission:

| Role | Name | Contact |
|---|---|---|
| Student(s) | | |
| Project guide | | |
| College / department | | |

To show an email address on the privacy and terms pages, set the environment variable `FOOTBALL_CONTACT_EMAIL`.

## 18. Further documents

* `docs/ALGORITHMS.md`: simple explanations of every algorithm used
* `docs/VIVA_QUESTIONS.md`: 30 viva questions with answers
* `docs/PRESENTATION_OUTLINE.md`: slide-by-slide presentation outline
* `docs/DEMO_SCRIPT.md`: a five-minute demonstration script, including the sample's calibration points
* `docs/NOVELTY_AND_APPLICATIONS.md`: what is new in this project and where it can be used

## Licence

This project is released under the **GNU Affero General Public License v3.0** (see [LICENSE](LICENSE)), the same
licence as Ultralytics YOLO, which it uses. You may use, study, change and share the code, but if you distribute a
modified version, or let others use it over a network (for example as a hosted website), you must make your
complete source code available to them under the same licence. The other libraries keep their own licences (listed
on the Terms of use page in the app). The sample video and the model weights are not part of this repository and
are covered by their own terms.
