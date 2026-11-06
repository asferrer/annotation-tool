"""
Videos Router - video upload (multipart streaming) + proxy to segmentation.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from app.proxy import proxy_segmentation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Videos"], prefix="/videos")

VIDEOS_BASE = Path(os.environ.get("VIDEOS_BASE", "/app/datasets/videos"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
UPLOAD_CHUNK = 4 * 1024 * 1024


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Stream a video upload into the datasets volume. Returns its video_id."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(422, f"Unsupported video extension: {suffix}")

    video_id = uuid.uuid4().hex[:12]
    video_dir = VIDEOS_BASE / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"source{suffix}"

    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK):
            out.write(chunk)
            size += len(chunk)

    logger.info(f"Video uploaded: {dest} ({size / 1e6:.1f} MB)")
    return {
        "video_id": video_id,
        "video_path": str(dest),
        "frames_dir": str(video_dir / "frames"),
        "size_bytes": size,
        "original_name": file.filename,
    }


@router.post("/probe")
async def probe(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/video/probe", body)


@router.post("/extract-frames")
async def extract_frames(body: Dict[str, Any] = Body(...)) -> Any:
    # Frame extraction of long videos can take a while
    return await proxy_segmentation("POST", "/video/extract-frames", body, timeout=600.0)


@router.post("/track")
async def track(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/video/track", body)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Any:
    return await proxy_segmentation("GET", f"/video/jobs/{job_id}")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> Any:
    return await proxy_segmentation("DELETE", f"/video/jobs/{job_id}")


@router.get("/models/status")
async def models_status() -> Any:
    return await proxy_segmentation("GET", "/models/status")
