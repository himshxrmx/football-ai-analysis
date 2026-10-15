"""Central configuration for the Football Video Analysis System.

Every tunable value used by the processing modules and the Streamlit
interface lives here so it can be changed in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"

for _directory in (SAMPLES_DIR, UPLOADS_DIR, MODELS_DIR, OUTPUTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Video validation
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MAX_FILE_SIZE_MB = 500
MAX_DURATION_SECONDS = 600  # 10 minutes
MIN_DURATION_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------
# Official Ultralytics COCO weights. They are downloaded automatically from
# the Ultralytics GitHub release page into MODELS_DIR on first use.
PRETRAINED_MODELS = {
    "yolo26n.pt": "YOLO26 nano (fastest, recommended for CPU)",
    "yolo26s.pt": "YOLO26 small (more accurate, about 2 to 3 times slower)",
    "yolo11n.pt": "YOLO11 nano (previous generation, for comparison)",
}
DEFAULT_MODEL = "yolo26n.pt"
POSE_MODEL = "yolo26n-pose.pt"      # optional pose page, also downloaded on first use

# Optional football-specific weights (classes such as player, goalkeeper,
# referee, ball). Put a fine-tuned file at models/football.pt or point the
# FOOTBALL_MODEL_PATH environment variable at it.
CUSTOM_MODEL_PATH = Path(os.environ.get("FOOTBALL_MODEL_PATH", MODELS_DIR / "football.pt"))

DEVICE = os.environ.get("FOOTBALL_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Processing defaults (all can be changed in the interface)
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE = 0.30
IMAGE_SIZES = (640, 960, 1280)
DEFAULT_IMAGE_SIZE = 960
DEFAULT_FRAME_STRIDE = 2
DEFAULT_MAX_SECONDS = 30
DEFAULT_TRAIL_LENGTH = 25           # processed frames shown in a movement trail
OUTPUT_MAX_WIDTH = 1280             # width of the annotated output video
TRACK_LOST_SECONDS = 2.0            # how long ByteTrack keeps a lost track alive
TRACKER_LOW_CONFIDENCE = 0.10       # ByteTrack second-stage association threshold
# ByteTrack accepts a match when 1 - IoU is below this value. Players in wide
# broadcast shots are small and move far between analysed frames, so their boxes
# overlap little; 0.9 (IoU >= 0.1) cut the number of track IDs on the 30 s sample
# clip from 110 (default 0.8) to 49 for the same detections. BoT-SORT with camera
# motion compensation was also tested and produced 55 IDs at 15 times the cost.
TRACKER_MATCH_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
MIN_TRACK_SECONDS = 2.0             # shorter tracks are excluded from player statistics
SMOOTHING_WINDOW = 5                # rolling window (processed frames) for positions
MAX_GAP_FRAMES = 3                  # larger gaps inside a track are not counted as movement
MAX_HUMAN_SPEED_MPS = 12.0          # faster steps are treated as tracking errors
HIGH_SPEED_THRESHOLD_MPS = 5.5      # 19.8 km/h, common high-speed running threshold
SPRINT_THRESHOLD_MPS = 7.0          # 25.2 km/h, common sprint threshold
MIN_BURST_SECONDS = 1.0

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# ---------------------------------------------------------------------------
# Optional language model for the coaching summary (disabled by default).
# Only a local Ollama server is supported so the project never needs a paid
# API key. Example: set COACH_LLM_ENABLED=1 and run `ollama run llama3.2`.
# ---------------------------------------------------------------------------
COACH_LLM_ENABLED = os.environ.get("COACH_LLM_ENABLED", "0") == "1"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

APP_NAME = "Football Video Analysis"
APP_VERSION = "1.0.0"
# Shown on the privacy and terms pages. Fill in before hosting the app publicly.
CONTACT_EMAIL = os.environ.get("FOOTBALL_CONTACT_EMAIL", "")
