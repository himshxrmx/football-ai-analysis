"""Video validation, reading and writing.

OpenCV is used to read videos frame by frame. Annotated videos are written
with H.264 through the ffmpeg binary that ships with the imageio-ffmpeg
package, because the mp4v codec produced by plain OpenCV does not play in
web browsers. If ffmpeg is unavailable the writer falls back to OpenCV.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config

try:  # imageio-ffmpeg bundles a static ffmpeg executable
    import imageio_ffmpeg

    FFMPEG_EXE: str | None = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover - depends on the environment
    imageio_ffmpeg = None
    FFMPEG_EXE = shutil.which("ffmpeg")


class VideoError(Exception):
    """Raised when a video cannot be opened or read."""


@dataclass
class VideoInfo:
    path: str
    file_name: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    size_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: VideoInfo | None = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_extension(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in config.ALLOWED_EXTENSIONS


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    text = "".join(chars).strip("\x00 ")
    return text if text.isprintable() and text else "unknown"


def get_video_info(path: str | Path) -> VideoInfo:
    """Read basic properties of a video. Raises VideoError if unreadable."""
    path = Path(path).resolve()
    if not path.exists():
        raise VideoError(f"File not found: {path.name}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV could not open the file. It may be corrupted or use an unsupported codec.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        codec = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(path),
        file_name=path.name,
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_s=round(duration, 2),
        codec=codec,
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
    )


def validate_video_file(
    path: str | Path,
    max_size_mb: float = config.MAX_FILE_SIZE_MB,
    max_duration_s: float = config.MAX_DURATION_SECONDS,
    min_duration_s: float = config.MIN_DURATION_SECONDS,
) -> ValidationResult:
    """Check file type, size, duration and that frames can be decoded."""
    path = Path(path)
    result = ValidationResult(ok=False)

    if not validate_extension(path.name):
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        result.errors.append(f"Unsupported file type '{path.suffix}'. Allowed types: {allowed}.")
        return result
    if not path.exists():
        result.errors.append("The file does not exist.")
        return result
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        result.errors.append("The file is empty.")
        return result
    if size_mb > max_size_mb:
        result.errors.append(f"The file is {size_mb:.1f} MB. The limit is {max_size_mb} MB.")
        return result

    try:
        info = get_video_info(path)
    except VideoError as exc:
        result.errors.append(str(exc))
        return result

    if info.width <= 0 or info.height <= 0:
        result.errors.append("The video reports an invalid frame size.")
    if info.fps <= 0 or info.fps > 240:
        result.errors.append(f"The video reports an invalid frame rate ({info.fps}).")
    if info.frame_count <= 0:
        result.errors.append("The video reports no frames.")
    if result.errors:
        return result

    # Decode the first frame and one frame from the middle to prove the file is readable.
    cap = cv2.VideoCapture(str(path))
    try:
        ok_first, first = cap.read()
        if not ok_first or first is None:
            result.errors.append("The first frame could not be decoded. The file may be corrupted.")
            return result
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, info.frame_count // 2))
        ok_mid, _ = cap.read()
        if not ok_mid:
            result.warnings.append("A frame in the middle of the video could not be decoded. Processing may stop early.")
    finally:
        cap.release()

    if info.duration_s < min_duration_s:
        result.errors.append(f"The video is {info.duration_s:.1f} s long. At least {min_duration_s:.0f} s is required.")
        return result
    if info.duration_s > max_duration_s:
        result.errors.append(
            f"The video is {info.duration_s / 60:.1f} minutes long. The limit is {max_duration_s / 60:.0f} minutes. "
            "Trim it to a shorter clip first."
        )
        return result
    if info.width < 320 or info.height < 240:
        result.warnings.append("The resolution is very low. Players may be too small to detect reliably.")
    if info.duration_s > 120:
        result.warnings.append("Long videos take a long time on a CPU. Consider limiting the processed duration.")

    result.ok = True
    result.info = info
    return result


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def safe_file_name(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "video"
    return f"{stem[:60]}{suffix}"


def save_uploaded_bytes(data: bytes, original_name: str, dest_dir: Path = config.UPLOADS_DIR) -> Path:
    """Store an uploaded file in its own time-stamped folder, keeping a filesystem-safe version of its name."""
    folder = dest_dir / time.strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while folder.exists():
        folder = dest_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    target = folder / safe_file_name(original_name)
    target.write_bytes(data)
    return target


def list_sample_videos(sample_dir: Path = config.SAMPLES_DIR) -> list[Path]:
    if not sample_dir.exists():
        return []
    return sorted(p for p in sample_dir.iterdir() if p.is_file() and validate_extension(p.name))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def iter_frames(
    path: str | Path,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame_bgr) for every `stride`-th frame.

    `max_frames` limits how many source frames are scanned (not how many are yielded).
    Skipped frames are grabbed without decoding, which is much faster than reading them.
    """
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open {Path(path).name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    index = 0
    try:
        while max_frames is None or index < max_frames:
            if index % stride == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield index, index / fps, frame
            else:
                if not cap.grab():
                    break
            index += 1
    finally:
        cap.release()


def read_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Return one frame (BGR). Raises VideoError if it cannot be decoded."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Could not open {Path(path).name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"Frame {frame_index} could not be decoded.")
        return frame
    finally:
        cap.release()


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
class VideoWriter:
    """Write BGR frames to an MP4 file that plays in browsers (H.264 when possible)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.fps = max(1.0, float(fps))
        self.width, self.height = size
        # libx264 with yuv420p needs even dimensions.
        self.out_width = self.width - (self.width % 2)
        self.out_height = self.height - (self.height % 2)
        self.backend = "none"
        self._ffmpeg = None
        self._cv_writer = None
        self.frames_written = 0
        self._open()

    def _open(self) -> None:
        if imageio_ffmpeg is not None:
            try:
                self._ffmpeg = imageio_ffmpeg.write_frames(
                    str(self.path),
                    (self.out_width, self.out_height),
                    fps=self.fps,
                    codec="libx264",
                    pix_fmt_in="bgr24",
                    pix_fmt_out="yuv420p",
                    quality=None,
                    bitrate=None,
                    macro_block_size=2,
                    output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"],
                    ffmpeg_log_level="error",
                )
                self._ffmpeg.send(None)  # start the generator
                self.backend = "ffmpeg-h264"
                return
            except Exception:
                self._ffmpeg = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.out_width, self.out_height))
        if not self._cv_writer.isOpened():
            raise VideoError("Could not create the output video file.")
        self.backend = "opencv-mp4v"

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.out_width or frame.shape[0] != self.out_height:
            frame = frame[: self.out_height, : self.out_width]
        if self._ffmpeg is not None:
            self._ffmpeg.send(np.ascontiguousarray(frame))
        else:
            self._cv_writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._ffmpeg is not None:
            self._ffmpeg.close()
            self._ffmpeg = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def is_browser_playable(info: VideoInfo) -> bool:
    return Path(info.path).suffix.lower() == ".mp4" and info.codec.lower() in {"avc1", "h264", "x264"}


def make_browser_preview(src: str | Path, dest: str | Path, max_seconds: float | None = None) -> Path | None:
    """Transcode a video to H.264 MP4 for playback in the browser. Returns None if ffmpeg is missing."""
    if FFMPEG_EXE is None:
        return None
    dest = Path(dest)
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", str(src)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-an",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError):
        return None
    return dest if dest.exists() and dest.stat().st_size > 0 else None
