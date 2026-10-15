"""Input validation, reading and writing of videos."""
from __future__ import annotations

import cv2
import numpy as np

from modules import video_processor as vp
from tests.conftest import write_video


def test_extension_check():
    assert vp.validate_extension("match.MP4")
    assert vp.validate_extension("clip.avi")
    assert not vp.validate_extension("notes.txt")
    assert not vp.validate_extension("video")


def test_unsupported_type_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    result = vp.validate_video_file(path)
    assert not result.ok
    assert "Unsupported file type" in result.errors[0]


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    result = vp.validate_video_file(path)
    assert not result.ok and "empty" in result.errors[0]


def test_missing_file_is_rejected(tmp_path):
    result = vp.validate_video_file(tmp_path / "nope.mp4")
    assert not result.ok


def test_corrupted_file_is_rejected(corrupted_video):
    result = vp.validate_video_file(corrupted_video)
    assert not result.ok
    assert result.errors


def test_size_limit(synthetic_video):
    result = vp.validate_video_file(synthetic_video, max_size_mb=0.0001)
    assert not result.ok and "limit" in result.errors[0]


def test_too_short_video_is_rejected(tmp_path):
    path = write_video(tmp_path / "short.mp4", frames=5, fps=10)
    result = vp.validate_video_file(path, min_duration_s=1.0)
    assert not result.ok and "At least" in result.errors[0]


def test_valid_video_info(synthetic_video):
    result = vp.validate_video_file(synthetic_video)
    assert result.ok, result.errors
    info = result.info
    assert (info.width, info.height) == (320, 240)
    assert abs(info.fps - 10.0) < 0.01
    assert info.frame_count == 30
    assert abs(info.duration_s - 3.0) < 0.01


def test_iter_frames_with_stride(synthetic_video):
    frames = list(vp.iter_frames(synthetic_video, stride=3))
    assert [f[0] for f in frames] == list(range(0, 30, 3))
    assert abs(frames[1][1] - 0.3) < 1e-6
    assert frames[0][2].shape == (240, 320, 3)


def test_iter_frames_max_frames(synthetic_video):
    frames = list(vp.iter_frames(synthetic_video, stride=2, max_frames=10))
    assert [f[0] for f in frames] == [0, 2, 4, 6, 8]


def test_read_frame(synthetic_video):
    frame = vp.read_frame(synthetic_video, 5)
    assert frame.shape == (240, 320, 3)


def test_writer_produces_readable_video(tmp_path):
    out = tmp_path / "out.mp4"
    with vp.VideoWriter(out, 10, (321, 241)) as writer:  # odd size must be handled
        for _ in range(12):
            writer.write(np.full((241, 321, 3), 128, dtype=np.uint8))
    assert writer.frames_written == 12
    cap = cv2.VideoCapture(str(out))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 12


def test_writer_uses_browser_codec_when_ffmpeg_available(tmp_path):
    out = tmp_path / "h264.mp4"
    with vp.VideoWriter(out, 10, (320, 240)) as writer:
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    if vp.imageio_ffmpeg is not None:
        assert writer.backend == "ffmpeg-h264"
        assert vp.is_browser_playable(vp.get_video_info(out))


def test_safe_file_name_and_upload(tmp_path):
    assert vp.safe_file_name("My Match (final).MP4") == "My_Match_final.mp4"
    saved = vp.save_uploaded_bytes(b"abc", "a b.mp4", tmp_path)
    again = vp.save_uploaded_bytes(b"xyz", "a b.mp4", tmp_path)
    assert saved.exists() and saved.read_bytes() == b"abc"
    assert saved.name == "a_b.mp4" and again.name == "a_b.mp4"
    assert saved.parent != again.parent  # two uploads with the same name never overwrite each other
