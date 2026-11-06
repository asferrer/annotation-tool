"""
Frame extraction from videos using OpenCV (bundled ffmpeg codecs).

Frames are written as JPEG q=90 named %06d.jpg so they sort naturally and
can be fed to Sam3VideoProcessor.init_video_session as an image folder.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2

from .schemas import VideoInfo

logger = logging.getLogger(__name__)

VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "3000"))
JPEG_QUALITY = 90


def probe(video_path: str) -> VideoInfo:
    """Read container metadata without decoding the whole video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video (unsupported codec?): {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(path), width=width, height=height,
        fps=round(fps, 3), frame_count=frame_count, duration_s=round(duration, 2),
    )


def _compute_stride(
    info: VideoInfo,
    target_fps: Optional[float],
    frame_stride: Optional[int],
    max_frames: int,
) -> int:
    if frame_stride:
        stride = frame_stride
    elif target_fps and info.fps > 0:
        stride = max(1, round(info.fps / target_fps))
    else:
        stride = 1
    # Never extract more than max_frames, whatever the requested sampling
    if info.frame_count > 0:
        min_stride = -(-info.frame_count // max_frames)  # ceil division
        stride = max(stride, min_stride)
    return stride


def extract(
    video_path: str,
    out_dir: str,
    target_fps: Optional[float] = None,
    frame_stride: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[VideoInfo, Path, int, int]:
    """Extract frames. Returns (video_info, frames_dir, extracted_count, stride)."""
    info = probe(video_path)
    limit = min(max_frames or VIDEO_MAX_FRAMES, VIDEO_MAX_FRAMES)
    stride = _compute_stride(info, target_fps, frame_stride, limit)

    frames_dir = Path(out_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(video_path)
    extracted = 0
    frame_idx = 0
    try:
        while extracted < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                out_file = frames_dir / f"{extracted:06d}.jpg"
                cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                extracted += 1
            frame_idx += 1
    finally:
        cap.release()

    logger.info(
        f"Extracted {extracted} frames from {video_path} "
        f"(stride={stride}, source={info.frame_count} frames)"
    )
    return info, frames_dir, extracted, stride
