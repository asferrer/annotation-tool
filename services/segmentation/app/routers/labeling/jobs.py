"""
Job queries and lifecycle: list, status, results, previews, cancel and delete.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import LabelingJobStatus, LabelingResultResponse, JobStatus
from app.service_state import state, labeling_jobs

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/labeling/jobs", tags=["Labeling Tool"])
async def list_labeling_jobs():
    """List all labeling jobs (from memory and database)."""
    jobs = []
    seen_job_ids = set()

    # First, get jobs from memory (these are the most up-to-date)
    for job_id, job in labeling_jobs.items():
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Check if job can be resumed (has checkpoint and is failed/cancelled)
        can_resume = False
        if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
            output_dir = Path(job.get("output_dir", ""))
            checkpoint_path = output_dir / "checkpoint.json"
            can_resume = checkpoint_path.exists()

        # Calculate progress
        total_images = job.get("total_images", 0)
        processed_images = job.get("processed_images", 0)
        progress = round((processed_images / total_images * 100), 1) if total_images > 0 else 0.0

        jobs.append({
            "job_id": job_id,
            "type": "labeling",  # Frontend expects 'type' field
            "job_type": job.get("job_type", "labeling"),
            "status": status_str,
            "progress": progress,  # Add progress percentage
            "created_at": job.get("created_at", job.get("started_at", datetime.now().isoformat())),
            "total_images": total_images,
            "processed_images": processed_images,
            "total_objects_found": job.get("total_objects_found", 0),
            "objects_by_class": job.get("objects_by_class", {}),
            "output_dir": job.get("output_dir", ""),
            "image_directories": job.get("image_directories", []),
            "current_image": job.get("current_image", ""),  # For progress display
            "errors": job.get("errors", [])[:5],  # Include first 5 errors
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
            "can_resume": can_resume,
        })
        seen_job_ids.add(job_id)

    # Then, add jobs from database that are not in memory (e.g., interrupted jobs from previous runs)
    if state.db:
        try:
            db_jobs = state.db.list_jobs(service="segmentation", job_type="labeling", limit=50)
            for db_job in db_jobs:
                job_id = db_job.get("id")
                if job_id and job_id not in seen_job_ids:
                    # This job exists in DB but not in memory - likely from a previous run
                    output_path = db_job.get("output_path", "")
                    can_resume = False
                    if db_job.get("status") in ["interrupted", "failed"]:
                        checkpoint_path = Path(output_path) / "checkpoint.json"
                        can_resume = checkpoint_path.exists()

                    # Parse result_summary if available
                    result_summary = db_job.get("result_summary", {}) or {}

                    # Calculate progress
                    total_images = db_job.get("total_items", 0)
                    processed_images = db_job.get("processed_items", 0)
                    progress = round((processed_images / total_images * 100), 1) if total_images > 0 else 0.0

                    jobs.append({
                        "job_id": job_id,
                        "type": "labeling",  # Frontend expects 'type' field
                        "job_type": "labeling",
                        "status": db_job.get("status", "unknown"),
                        "progress": progress,  # Add progress percentage
                        "created_at": db_job.get("created_at", db_job.get("started_at", datetime.now().isoformat())),
                        "total_images": total_images,
                        "processed_images": processed_images,
                        "total_objects_found": result_summary.get("total_objects_found", 0),
                        "objects_by_class": result_summary.get("objects_by_class", {}),
                        "output_dir": output_path,
                        "image_directories": (db_job.get("request_params") or {}).get("image_directories", []),
                        "current_image": db_job.get("current_item", ""),
                        "errors": [],
                        "started_at": db_job.get("started_at"),
                        "completed_at": db_job.get("completed_at"),
                        "processing_time_ms": db_job.get("processing_time_ms", 0),
                        "can_resume": can_resume,
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch jobs from database: {e}")

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/labeling/jobs/{job_id}", response_model=LabelingJobStatus, tags=["Labeling Tool"])
async def get_labeling_job_status(job_id: str):
    """Get status of a labeling job."""
    if job_id not in labeling_jobs:
        # Fallback: reconstruct status from database for jobs from previous service runs
        if state.db:
            try:
                db_job = state.db.get_job(job_id)
                if db_job:
                    total_images = db_job.get("total_items", 0)
                    processed_images = db_job.get("processed_items", 0)
                    progress = round((processed_images / total_images * 100), 1) if total_images > 0 else 0.0
                    result_summary = db_job.get("result_summary", {}) or {}
                    output_path = db_job.get("output_path", "")
                    can_resume = False
                    if db_job.get("status") in ["interrupted", "failed"]:
                        can_resume = (Path(output_path) / "checkpoint.json").exists()
                    req_params = db_job.get("request_params") or {}
                    return LabelingJobStatus(
                        job_id=job_id,
                        job_type=db_job.get("job_type", "labeling"),
                        status=db_job.get("status", "unknown"),
                        total_images=total_images,
                        processed_images=processed_images,
                        progress=progress,
                        annotations_created=result_summary.get("total_objects_found", 0),
                        total_objects_found=result_summary.get("total_objects_found", 0),
                        objects_by_class=result_summary.get("objects_by_class", {}),
                        output_dir=output_path,
                        image_directories=req_params.get("image_directories", []),
                        started_at=db_job.get("started_at"),
                        completed_at=db_job.get("completed_at"),
                        can_resume=can_resume,
                    )
            except Exception as e:
                logger.warning(f"DB fallback failed for job {job_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    # Check if job can be resumed (has checkpoint and is failed/cancelled)
    can_resume = False
    status = job.get("status")
    if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
        output_dir = Path(job.get("output_dir", ""))
        checkpoint_path = output_dir / "checkpoint.json"
        can_resume = checkpoint_path.exists()

    # Calculate progress percentage
    total_images = job.get("total_images", 0)
    processed_images = job.get("processed_images", 0)
    progress = (processed_images / total_images * 100) if total_images > 0 else 0.0

    # Get annotations count
    total_objects = job.get("total_objects_found", 0)

    # Build quality metrics if available
    quality_metrics = None
    if "quality_metrics" in job:
        from app.models.extraction_schemas import LabelingQualityMetrics
        qm = job["quality_metrics"]
        quality_metrics = LabelingQualityMetrics(
            avg_confidence=qm.get("avg_confidence", 0.0),
            images_with_detections=qm.get("images_with_detections", 0),
            images_without_detections=qm.get("images_without_detections", 0),
            low_confidence_count=qm.get("low_confidence_count", 0),
            total_detections=qm.get("total_detections", 0),
        )

    return LabelingJobStatus(
        job_id=job_id,
        job_type=job.get("job_type", "labeling"),
        status=status or JobStatus.QUEUED,
        total_images=total_images,
        processed_images=processed_images,
        progress=round(progress, 1),  # Percentage of completion
        annotations_created=total_objects,  # Frontend expects this name
        total_objects_found=total_objects,  # Keep for backwards compatibility
        objects_by_class=job.get("objects_by_class", {}),
        current_image=job.get("current_image", ""),
        output_dir=job.get("output_dir", ""),
        image_directories=job.get("image_directories", []),
        output_formats=job.get("output_formats", []),
        errors=job.get("errors", [])[:50],
        warnings=job.get("warnings", [])[:20],  # Include warnings
        quality_metrics=quality_metrics,  # Include quality metrics
        processing_time_ms=job.get("processing_time_ms", 0),
        created_at=job.get("created_at"),  # When job was created
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        can_resume=can_resume,
    )


@router.get("/labeling/jobs/{job_id}/result", response_model=LabelingResultResponse, tags=["Labeling Tool"])
async def get_labeling_result(job_id: str):
    """Get the result of a completed labeling job."""
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    if job.get("status") != JobStatus.COMPLETED:
        return LabelingResultResponse(
            success=False,
            error=f"Job is not completed. Current status: {job.get('status')}"
        )

    # Load the COCO result
    output_dir = Path(job.get("output_dir", ""))
    coco_path = output_dir / "annotations.json"

    if not coco_path.exists():
        return LabelingResultResponse(
            success=False,
            error="Result file not found"
        )

    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        # Build output files map
        output_files = {"coco": str(coco_path)}

        yolo_dir = output_dir / "yolo"
        if yolo_dir.exists():
            output_files["yolo"] = str(yolo_dir)

        voc_dir = output_dir / "voc"
        if voc_dir.exists():
            output_files["voc"] = str(voc_dir)

        return LabelingResultResponse(
            success=True,
            data=coco_data,
            output_files=output_files,
            summary={
                "total_images": len(coco_data.get("images", [])),
                "total_annotations": len(coco_data.get("annotations", [])),
                "categories": [c.get("name") for c in coco_data.get("categories", [])],
            }
        )

    except Exception as e:
        return LabelingResultResponse(
            success=False,
            error=str(e)
        )


@router.get("/labeling/jobs/{job_id}/partial-annotations", tags=["Labeling Tool"])
async def get_partial_annotations(job_id: str, max_images: int = 0):
    """Get partial COCO annotations for a running labeling job.

    Returns all processed images by default (max_images=0 means no limit).
    Pass max_images > 0 to cap the response to the last N images.
    """
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]
    coco_data = job.get("_coco_result")

    if coco_data is None:
        # Fallback to checkpoint file
        output_dir = Path(job.get("output_dir", ""))
        checkpoint_path = output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                coco_data = checkpoint.get("coco_data")
            except Exception:
                pass

    if coco_data is None:
        return {
            "job_id": job_id,
            "available": False,
            "processed_images": job.get("processed_images", 0),
            "total_images": job.get("total_images", 0),
        }

    # Build filename -> full path map from job's image paths
    image_path_map: Dict[str, str] = {}
    for p in job.get("_image_paths", []):
        image_path_map[Path(p).name] = p

    # Enrich images with full paths for frontend image serving
    all_images = []
    for img in coco_data.get("images", []):
        enriched = dict(img)
        enriched["full_path"] = image_path_map.get(img.get("file_name", ""), "")
        all_images.append(enriched)

    # Return all images by default; max_images > 0 caps to last N
    enriched_images = all_images[-max_images:] if max_images > 0 and len(all_images) > max_images else all_images
    recent_image_ids = {img["id"] for img in enriched_images}

    # Strip internal fields and filter to recent images only
    clean_annotations = []
    for ann in coco_data.get("annotations", []):
        if ann.get("image_id") not in recent_image_ids:
            continue
        clean_ann = {k: v for k, v in ann.items() if not k.startswith("_")}
        clean_annotations.append(clean_ann)

    return {
        "job_id": job_id,
        "available": True,
        "data": {
            "images": enriched_images,
            "annotations": clean_annotations,
            "categories": coco_data.get("categories", []),
        },
        "processed_images": job.get("processed_images", 0),
        "total_images": job.get("total_images", 0),
    }


@router.get("/labeling/jobs/{job_id}/previews", tags=["Labeling Tool"])
async def get_labeling_job_previews(job_id: str, limit: int = 10):
    """
    Get preview images for a labeling job.

    Returns base64-encoded preview images showing the annotations in progress.
    Useful for monitoring the quality of auto-labeling in real-time.
    """
    import base64

    # First check memory
    job = labeling_jobs.get(job_id)
    output_dir = None

    if job:
        output_dir = Path(job.get("output_dir", ""))
    elif state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            output_dir = Path(db_job.get("output_path", ""))

    if output_dir is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    previews_dir = output_dir / "previews"
    if not previews_dir.exists():
        return {
            "job_id": job_id,
            "previews": [],
            "total": 0,
            "message": "No preview images available yet"
        }

    # Get preview files sorted by modification time (most recent first)
    preview_files = sorted(
        previews_dir.glob("preview_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    previews = []
    for preview_file in preview_files:
        try:
            with open(preview_file, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            previews.append({
                "filename": preview_file.name,
                "path": str(preview_file),
                "image_data": img_data,  # Base64 without prefix
                "timestamp": str(preview_file.stat().st_mtime),
                "size_kb": preview_file.stat().st_size / 1024,
            })
        except Exception as e:
            logger.warning(f"Failed to read preview {preview_file}: {e}")

    return {
        "job_id": job_id,
        "previews": previews,
        "total": len(preview_files),
        "output_dir": str(output_dir),
    }


@router.delete("/labeling/jobs/{job_id}", tags=["Labeling Tool"])
async def cancel_labeling_job(job_id: str):
    """
    Cancel a running labeling job.

    Sets the job status to CANCELLED and allows it to be resumed later
    from the checkpoint.
    """
    # Check memory first
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        status = job.get("status")

        # Can only cancel running/processing/queued jobs
        if status in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            job["status"] = JobStatus.CANCELLED
            job["completed_at"] = datetime.now().isoformat()

            # Update database if available
            if state.db:
                try:
                    state.db.update_job_status(job_id, "cancelled")
                except Exception as e:
                    logger.warning(f"Failed to update job status in database: {e}")

            logger.info(f"Cancelled labeling job {job_id}")
            return {
                "success": True,
                "job_id": job_id,
                "message": "Job cancelled successfully. Can be resumed from checkpoint.",
                "processed_images": job.get("processed_images", 0),
                "total_images": job.get("total_images", 0),
            }
        else:
            status_str = status.value if hasattr(status, 'value') else str(status)
            return {
                "success": False,
                "error": f"Job cannot be cancelled. Current status: {status_str}"
            }

    # Check database
    if state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            if db_job.get("status") in ["running", "pending"]:
                state.db.update_job_status(job_id, "cancelled")
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Job cancelled in database."
                }
            else:
                return {
                    "success": False,
                    "error": f"Job cannot be cancelled. Current status: {db_job.get('status')}"
                }

    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/labeling/jobs/{job_id}/delete", tags=["Labeling Tool"])
async def delete_labeling_job(job_id: str, delete_files: bool = False):
    """
    Delete a labeling job from memory and database.

    Args:
        job_id: The job ID to delete
        delete_files: If True, also delete output files (default: False)

    Note: If the job is running, it will be cancelled first.
    """
    output_dir = None
    was_cancelled = False

    # First, try to cancel if running
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        output_dir = job.get("output_dir")

        if job.get("status") in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            job["status"] = JobStatus.CANCELLED
            job["completed_at"] = datetime.now().isoformat()
            was_cancelled = True
            logger.info(f"Cancelled running labeling job {job_id} before deletion")

        # Remove from memory
        del labeling_jobs[job_id]
        logger.info(f"Removed labeling job {job_id} from memory")

    # Remove from database
    db_deleted = False
    if state.db:
        try:
            db_job = state.db.get_job(job_id)
            if db_job:
                if output_dir is None:
                    output_dir = db_job.get("output_path")
                state.db.delete_job(job_id)
                db_deleted = True
                logger.info(f"Removed labeling job {job_id} from database")
        except Exception as e:
            logger.warning(f"Failed to delete job from database: {e}")

    # Delete files if requested
    files_deleted = False
    if delete_files and output_dir:
        try:
            output_path = Path(output_dir)
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
                files_deleted = True
                logger.info(f"Deleted output directory: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete output files: {e}")

    if job_id not in labeling_jobs and not db_deleted:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return {
        "success": True,
        "job_id": job_id,
        "message": "Job deleted successfully",
        "details": {
            "was_cancelled": was_cancelled,
            "removed_from_memory": True,
            "removed_from_database": db_deleted,
            "files_deleted": files_deleted,
        }
    }
