"""
Resume endpoint for interrupted labeling jobs.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import LabelingJobResponse, JobStatus
from app.service_state import state, labeling_jobs
from app.routers.labeling.helpers import _get_image_paths_from_output_dir
from app.routers.labeling.processing import _process_labeling_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/labeling/jobs/{job_id}/resume", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def resume_labeling_job(job_id: str):
    """
    Resume a failed or interrupted labeling job.

    Loads the checkpoint and continues from the last processed image.
    Can resume jobs from database even after service restart.
    """
    job = None
    output_dir = None

    # First check memory
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        output_dir = Path(job.get("output_dir", ""))
    # If not in memory, try to load from database
    elif state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            output_dir = Path(db_job.get("output_path", ""))
            # Check if checkpoint exists before trying to reconstruct
            checkpoint_path = output_dir / "checkpoint.json"
            if not checkpoint_path.exists():
                return LabelingJobResponse(
                    success=False,
                    error="No checkpoint found. Job must be restarted from the beginning."
                )

            # Load checkpoint to get full job state
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)

                # Reconstruct job in memory from database and checkpoint
                request_params = db_job.get("request_params", {}) or {}
                labeling_jobs[job_id] = {
                    "job_id": job_id,
                    "job_type": "labeling",
                    "status": JobStatus.FAILED,  # Will be updated to PROCESSING
                    "total_images": db_job.get("total_items", 0),
                    "processed_images": checkpoint.get("last_processed_idx", 0),
                    "total_objects_found": sum(checkpoint.get("objects_by_class", {}).values()),
                    "objects_by_class": checkpoint.get("objects_by_class", {}),
                    "current_image": "",
                    "output_dir": str(output_dir),
                    "output_formats": request_params.get("output_formats", ["coco"]),
                    "errors": [],
                    "processing_time_ms": 0.0,
                    "started_at": None,
                    "completed_at": None,
                    # Reconstruct processing params from checkpoint/database
                    "_image_paths": _get_image_paths_from_output_dir(output_dir),
                    "_classes": request_params.get("classes", []),
                    "_task_type": request_params.get("task_type", "segmentation"),
                    "_min_confidence": request_params.get("min_confidence", 0.5),
                    "_min_area": 100,
                    "_max_instances": 100,
                    "_simplify_polygons": True,
                    "_simplify_tolerance": 2.0,
                    "_save_visualizations": True,  # Enable for resumed jobs
                    "_class_mapping": request_params.get("class_mapping"),
                    "_padding": 0,
                }
                job = labeling_jobs[job_id]
                logger.info(f"Reconstructed job {job_id} from database and checkpoint")
            except Exception as e:
                logger.error(f"Failed to reconstruct job from checkpoint: {e}")
                return LabelingJobResponse(
                    success=False,
                    error=f"Failed to reconstruct job: {str(e)}"
                )

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Only allow resuming failed/cancelled/interrupted jobs
    status = job.get("status")
    if hasattr(status, 'value'):
        status_str = status.value
    else:
        status_str = str(status)

    if status_str not in ["failed", "cancelled", "interrupted"]:
        return LabelingJobResponse(
            success=False,
            error=f"Job cannot be resumed. Current status: {status_str}. "
                  f"Only failed, cancelled, or interrupted jobs can be resumed."
        )

    # Check if checkpoint exists
    if output_dir is None:
        output_dir = Path(job.get("output_dir", ""))
    checkpoint_path = output_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        return LabelingJobResponse(
            success=False,
            error="No checkpoint found. Job must be restarted from the beginning."
        )

    # Load checkpoint to get resume point
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        resume_from = checkpoint.get("last_processed_idx", 0)
    except Exception as e:
        return LabelingJobResponse(
            success=False,
            error=f"Failed to load checkpoint: {str(e)}"
        )

    # Reset job status and start processing
    job["status"] = JobStatus.PROCESSING
    job["errors"] = []  # Clear previous errors
    job["started_at"] = datetime.now().isoformat()
    job["completed_at"] = None

    # Start background task to resume
    async def run_resume():
        try:
            await _process_labeling_job(job_id, resume_from=resume_from)
            labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
        except Exception as e:
            logger.exception(f"Resumed labeling job {job_id} failed: {e}")
            labeling_jobs[job_id]["status"] = JobStatus.FAILED
            labeling_jobs[job_id]["errors"].append(str(e))
        finally:
            labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    asyncio.create_task(run_resume())

    logger.info(f"Resumed labeling job {job_id} from image {resume_from}")

    return LabelingJobResponse(
        success=True,
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message=f"Job resumed from image {resume_from + 1} of {job.get('total_images', 0)}",
        total_images=job.get("total_images", 0)
    )
