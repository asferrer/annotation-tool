"""
Video endpoints: probe, frame extraction and SAM3 Video tracking jobs.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.model_manager import ModelBusyError, manager
from app.service_state import state, thread_pool
from app.video import frame_extractor
from app.video.schemas import (
    ExtractFramesRequest,
    ExtractFramesResponse,
    ProbeRequest,
    TrackJobResponse,
    TrackRequest,
    VideoInfo,
)
from app.video.tracker import cancel_flags, run_tracking_job

try:
    from shared.job_lifecycle import JobLifecycle
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, "/app")
    from shared.job_lifecycle import JobLifecycle

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models/status", tags=["Video"])
async def models_status():
    """Which engine currently owns the GPU (sam3 / sam3_video / insid3)."""
    return manager.status()


@router.post("/video/probe", response_model=VideoInfo, tags=["Video"])
async def probe_video(request: ProbeRequest):
    try:
        return await asyncio.to_thread(frame_extractor.probe, request.video_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/video/extract-frames", response_model=ExtractFramesResponse, tags=["Video"])
async def extract_frames(request: ExtractFramesRequest):
    out_dir = request.output_dir or str(Path(request.video_path).parent / "frames")
    try:
        info, frames_dir, extracted, stride = await asyncio.to_thread(
            frame_extractor.extract,
            request.video_path,
            out_dir,
            request.target_fps,
            request.frame_stride,
            request.max_frames,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

    effective_fps = info.fps / stride if info.fps > 0 else 0.0
    return ExtractFramesResponse(
        frames_dir=str(frames_dir),
        extracted_frames=extracted,
        frame_stride=stride,
        effective_fps=round(effective_fps, 3),
        video=info,
    )


@router.post("/video/track", response_model=TrackJobResponse, tags=["Video"])
async def start_tracking(request: TrackRequest):
    if not state.gpu_available:
        raise HTTPException(503, "GPU not available for video tracking")

    frames = sorted(Path(request.frames_dir).glob("*.jpg"))
    if not frames:
        raise HTTPException(404, f"No frames found in {request.frames_dir}")

    # Fail fast if another engine has active jobs (the worker would block otherwise)
    status = manager.status()
    active = status["active_model"]
    if active and active != "sam3_video" and status["refcounts"].get(active, 0) > 0:
        raise HTTPException(409, f"Model '{active}' has active jobs; retry when they finish")

    job_id = f"track_{uuid.uuid4().hex[:12]}"
    lifecycle = JobLifecycle.create(
        job_id, job_type="video_tracking", service="segmentation",
        request_params=request.model_dump(), total_items=len(frames),
        output_path=request.output_json,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(thread_pool, _run_job_safe, job_id, request, lifecycle)
    return TrackJobResponse(
        success=True, job_id=job_id, status="queued",
        message=f"Tracking {len(frames)} frames", total_frames=len(frames),
    )


def _run_job_safe(job_id: str, request: TrackRequest, lifecycle) -> None:
    try:
        run_tracking_job(job_id, request, lifecycle)
    except ModelBusyError as e:
        lifecycle.fail(str(e))
    except Exception as e:  # run_tracking_job already handles its own errors
        logger.exception(f"Unexpected tracking failure {job_id}")
        lifecycle.fail(str(e))


@router.get("/video/jobs/{job_id}", tags=["Video"])
async def get_video_job(job_id: str):
    if not state.db:
        raise HTTPException(503, "Job database unavailable")
    job = await asyncio.to_thread(state.db.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.delete("/video/jobs/{job_id}", tags=["Video"])
async def cancel_video_job(job_id: str):
    cancel_flags[job_id] = True
    return {"status": "cancel_requested", "job_id": job_id}
