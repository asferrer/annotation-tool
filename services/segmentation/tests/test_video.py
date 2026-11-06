"""Tests for the video pipeline (frame extraction + mask utils). CPU-only."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.video.frame_extractor import _compute_stride, extract, probe
from app.video.schemas import VideoInfo
from app.video.tracker import mask_to_polygons


@pytest.fixture()
def synthetic_video(tmp_path):
    """2-second synthetic video: 20 frames @ 10fps, 64x48."""
    path = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_probe(synthetic_video):
    info = probe(str(synthetic_video))
    assert info.width == 64 and info.height == 48
    assert info.frame_count == 20
    assert info.fps == pytest.approx(10.0, abs=0.5)


def test_probe_missing_file():
    with pytest.raises(FileNotFoundError):
        probe("/nonexistent/video.mp4")


def test_extract_all_frames(synthetic_video, tmp_path):
    info, frames_dir, extracted, stride = extract(
        str(synthetic_video), str(tmp_path / "frames")
    )
    assert extracted == 20 and stride == 1
    assert len(list(frames_dir.glob("*.jpg"))) == 20
    assert (frames_dir / "000000.jpg").exists()


def test_extract_with_target_fps(synthetic_video, tmp_path):
    _, _, extracted, stride = extract(
        str(synthetic_video), str(tmp_path / "frames"), target_fps=5
    )
    assert stride == 2
    assert extracted == 10


def test_extract_respects_max_frames(synthetic_video, tmp_path):
    _, _, extracted, stride = extract(
        str(synthetic_video), str(tmp_path / "frames"), max_frames=5
    )
    assert extracted <= 5
    assert stride >= 4


def test_compute_stride_caps_at_max_frames():
    info = VideoInfo(path="x", width=1, height=1, fps=30, frame_count=9000, duration_s=300)
    assert _compute_stride(info, None, None, max_frames=3000) == 3
    assert _compute_stride(info, target_fps=30, frame_stride=None, max_frames=100) == 90


def test_mask_to_polygons_square():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:40] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    coords = polygons[0]
    assert len(coords) >= 6
    xs, ys = coords[0::2], coords[1::2]
    assert min(xs) >= 9 and max(xs) <= 40
    assert min(ys) >= 9 and max(ys) <= 30


def test_mask_to_polygons_empty():
    assert mask_to_polygons(np.zeros((20, 20), dtype=bool)) == []
