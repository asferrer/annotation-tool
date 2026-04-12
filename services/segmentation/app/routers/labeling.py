"""
Labeling tool endpoints: auto-labeling, relabeling, job management.
"""

import os
import json
import time
import logging
import asyncio
import uuid
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import cv2
import numpy as np

from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import (
    StartLabelingRequest, StartRelabelingRequest,
    LabelingJobResponse, LabelingJobStatus, LabelingResultResponse,
    JobStatus,
)
from app.service_state import (
    state, labeling_jobs, wait_for_sam3,
    thread_pool, labeling_job_semaphore,
    MAX_CONCURRENT_LABELING_JOBS, MAX_CONCURRENT_IMAGES_PER_JOB,
    VRAM_MONITOR_AVAILABLE, VRAMMonitor,
    _calculate_iou,
)
from app.prompt_optimizer import get_prompt_optimizer
from app.detection_validator import get_detection_validator, deduplicate_annotations

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/labeling/start", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def start_labeling_job(request: StartLabelingRequest):
    """
    Start a new labeling job to label images from scratch.

    Uses SAM3 text prompts to detect and segment specified classes.
    Supports multiple image directories and output formats.
    """
    # Wait for SAM3 if still loading
    if state.sam3_loading:
        logger.info("Waiting for SAM3 to load for labeling job...")
        await wait_for_sam3(timeout=120.0)

    if not state.sam3_available:
        error_msg = state.sam3_load_error or "SAM3 not available. This feature requires SAM3 for text-based segmentation."
        return LabelingJobResponse(
            success=False,
            error=error_msg
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Validate directories and count images
        total_images = 0
        all_image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for dir_path in request.image_directories:
            dir_obj = Path(dir_path)
            if not dir_obj.exists():
                return LabelingJobResponse(
                    success=False,
                    error=f"Directory not found: {dir_path}"
                )

            for ext in image_extensions:
                for img_path in dir_obj.rglob(f"*{ext}"):
                    all_image_paths.append(str(img_path))
                for img_path in dir_obj.rglob(f"*{ext.upper()}"):
                    all_image_paths.append(str(img_path))

        # Deduplicate (case-insensitive FS may return same file for .jpg and .JPG)
        all_image_paths = list(dict.fromkeys(all_image_paths))
        total_images = len(all_image_paths)

        if total_images == 0:
            return LabelingJobResponse(
                success=False,
                error="No images found in specified directories"
            )

        # Handle preview mode (limit to N images for testing)
        if request.preview_mode:
            preview_limit = min(request.preview_count, total_images)
            all_image_paths = all_image_paths[:preview_limit]
            total_images = preview_limit
            logger.info(f"[LABEL] Preview mode enabled: processing first {total_images} images")
        else:
            # Warn about large datasets (may take hours/days to process)
            if total_images > 1000:
                logger.warning(f"[LABEL] Large dataset detected: {total_images} images. Processing may take several hours. Consider using preview mode first.")

        # Check for conflicting output directory with active jobs
        for existing_job in labeling_jobs.values():
            if (existing_job.get("status") in (JobStatus.QUEUED, JobStatus.PROCESSING)
                    and existing_job.get("output_dir") == request.output_dir):
                return LabelingJobResponse(
                    success=False,
                    error=f"A labeling job is already running/queued with the same output directory: {request.output_dir}"
                )

        # Create job with unique output subdirectory
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_subdir = f"labeling_{timestamp}_{job_id[:8]}"
        output_dir = Path(request.output_dir) / job_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LABEL] Job output directory: {output_dir}")

        # Determine final classes for tracking (use mapped names if mapping exists)
        if request.class_mapping:
            final_classes = list(dict.fromkeys(
                request.class_mapping.get(cls, cls) for cls in request.classes
            ))
        else:
            final_classes = request.classes

        labeling_jobs[job_id] = {
            "job_id": job_id,
            "job_type": "labeling",
            "status": JobStatus.QUEUED,
            "total_images": total_images,
            "processed_images": 0,
            "total_objects_found": 0,
            "objects_by_class": {cls: 0 for cls in final_classes},
            "current_image": "",
            "output_dir": str(output_dir),
            "output_formats": request.output_formats,
            "errors": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None,
            # Store request params for processing
            "_image_paths": all_image_paths,
            "_classes": request.classes,
            "_task_type": request.task_type,
            "_min_confidence": request.min_confidence,
            "_min_area": request.min_area,
            "_max_instances": request.max_instances_per_image,
            "_simplify_polygons": request.simplify_polygons,
            "_simplify_tolerance": request.simplify_tolerance,
            "_save_visualizations": request.save_visualizations,
            "_class_mapping": request.class_mapping,  # Maps prompts to final class names
            "_padding": request.padding,  # Pixels of padding around bboxes
            "_preview_mode": request.preview_mode,  # If True, this is a preview/test run
            "_deduplication_strategy": request.deduplication_strategy,  # Strategy for deduplication: confidence or area
        }

        # Persist job to database for durability
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="labeling",
                    service="segmentation",
                    request_params={
                        "classes": request.classes,
                        "class_mapping": request.class_mapping,
                        "output_formats": request.output_formats,
                        "task_type": request.task_type,
                        "min_confidence": request.min_confidence,
                        "image_directories": [str(d) for d in request.image_directories],
                    },
                    total_items=total_images,
                    output_path=str(output_dir)
                )
                logger.info(f"Labeling job {job_id} persisted to database")
            except Exception as e:
                logger.warning(f"Failed to persist job to database: {e}")

        # Start background task with concurrency control
        async def run_labeling():
            async with labeling_job_semaphore:
                logger.info(f"Labeling job {job_id} acquired semaphore (max {MAX_CONCURRENT_LABELING_JOBS} concurrent)")
                labeling_jobs[job_id]["status"] = JobStatus.PROCESSING
                labeling_jobs[job_id]["started_at"] = datetime.now().isoformat()

                # Update database status to running
                if state.db:
                    try:
                        state.db.update_job_status(job_id, "running", started_at=datetime.now())
                    except Exception as e:
                        logger.warning(f"Failed to update job status in database: {e}")

                try:
                    await _process_labeling_job(job_id)
                    labeling_jobs[job_id]["status"] = JobStatus.COMPLETED

                    # Update database with completion and record dataset metadata
                    if state.db:
                        try:
                            job = labeling_jobs[job_id]
                            output_path = job.get("output_dir", "")
                            coco_json_path = str(Path(output_path) / "annotations.json")
                            state.db.complete_job(
                                job_id=job_id,
                                status="completed",
                                result_summary={
                                    "total_objects_found": job.get("total_objects_found", 0),
                                    "objects_by_class": job.get("objects_by_class", {}),
                                    "output_path": output_path,
                                    "coco_json_path": coco_json_path,
                                },
                                processing_time_ms=job.get("processing_time_ms", 0)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update job completion in database: {e}")

                        # Save dataset metadata for completed labeling jobs
                        try:
                            job = labeling_jobs[job_id]
                            output_path = job.get("output_dir", "")
                            coco_json_path = str(Path(output_path) / "annotations.json")
                            if Path(coco_json_path).exists():
                                state.db.create_dataset_metadata(
                                    job_id=job_id,
                                    dataset_name=f"Labeled_{job_id[-8:]}",
                                    dataset_type="labeling",
                                    coco_json_path=coco_json_path,
                                    images_dir=output_path,
                                    num_images=job.get("total_images", 0),
                                    num_annotations=job.get("total_objects_found", 0),
                                    num_categories=len(job.get("objects_by_class", {})),
                                    class_distribution=job.get("objects_by_class", {}),
                                    categories=[
                                        {"id": i + 1, "name": cls}
                                        for i, cls in enumerate(job.get("objects_by_class", {}).keys())
                                    ],
                                )
                        except Exception as e:
                            logger.warning(f"Failed to save labeling dataset metadata: {e}")

                except Exception as e:
                    logger.exception(f"Labeling job {job_id} failed: {e}")
                    labeling_jobs[job_id]["status"] = JobStatus.FAILED
                    labeling_jobs[job_id]["errors"].append(str(e))

                    # Update database with failure
                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id=job_id,
                                status="failed",
                                error_message=str(e)
                            )
                        except Exception as db_err:
                            logger.warning(f"Failed to update job failure in database: {db_err}")

                finally:
                    labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    logger.info(f"Labeling job {job_id} released semaphore")

        asyncio.create_task(run_labeling())

        logger.info(f"Started labeling job {job_id} with {total_images} images, {len(request.classes)} classes")

        return LabelingJobResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Labeling job started. {total_images} images, {len(request.classes)} classes. Output: {output_dir}",
            total_images=total_images
        )

    except Exception as e:
        logger.error(f"Failed to start labeling job: {e}")
        return LabelingJobResponse(
            success=False,
            error=str(e)
        )


@router.post("/labeling/relabel", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def start_relabeling_job(request: StartRelabelingRequest):
    """
    Start a relabeling job for an existing dataset.

    Modes:
    - add: Add new class annotations while keeping existing ones
    - replace: Replace all annotations with new labeling
    - improve_segmentation: Convert bbox-only annotations to segmentations
    """
    # Wait for SAM3 if still loading
    if state.sam3_loading:
        logger.info("Waiting for SAM3 to load for relabeling job...")
        await wait_for_sam3(timeout=120.0)

    if not state.sam3_available:
        error_msg = state.sam3_load_error or "SAM3 not available. This feature requires SAM3."
        return LabelingJobResponse(
            success=False,
            error=error_msg
        )

    try:
        # Validate that annotation source is provided for relabeling
        if not request.coco_data and not request.coco_json_path:
            return LabelingJobResponse(
                success=False,
                error="Relabeling requires an existing annotations file. Please provide a COCO JSON path."
            )

        # If coco_json_path is provided but coco_data is not, read and parse the JSON file
        if request.coco_json_path and not request.coco_data:
            coco_path = Path(request.coco_json_path)
            if coco_path.exists():
                import json as json_module
                try:
                    with open(coco_path, "r", encoding="utf-8") as f:
                        request.coco_data = json_module.load(f)
                    logger.info(f"[RELABEL] Loaded COCO data from {coco_path}: "
                                f"{len(request.coco_data.get('images', []))} images, "
                                f"{len(request.coco_data.get('categories', []))} categories, "
                                f"{len(request.coco_data.get('annotations', []))} annotations")
                except Exception as e:
                    logger.error(f"[RELABEL] Failed to read COCO JSON from {coco_path}: {e}")
                    return LabelingJobResponse(
                        success=False,
                        error=f"Failed to read COCO JSON file: {e}"
                    )
            else:
                logger.warning(f"[RELABEL] COCO JSON path not found: {coco_path}")
                return LabelingJobResponse(
                    success=False,
                    error=f"COCO JSON file not found: {coco_path}"
                )

        # Get image paths from directories and/or existing dataset
        all_image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Build image lookup from directories (recursive search)
        image_lookup = {}      # basename -> full path
        image_lookup_rel = {}  # relative path -> full path
        for dir_path in request.image_directories:
            dir_obj = Path(dir_path)
            if dir_obj.exists():
                for ext in image_extensions:
                    for img_path in dir_obj.rglob(f"*{ext}"):
                        image_lookup[img_path.name] = str(img_path)
                        try:
                            rel = str(img_path.relative_to(dir_obj))
                            image_lookup_rel[rel] = str(img_path)
                        except ValueError:
                            pass
                    for img_path in dir_obj.rglob(f"*{ext.upper()}"):
                        image_lookup[img_path.name] = str(img_path)
                        try:
                            rel = str(img_path.relative_to(dir_obj))
                            image_lookup_rel[rel] = str(img_path)
                        except ValueError:
                            pass

        # Build image ID mapping from original COCO data (preserves original IDs)
        image_id_map = {}  # filename -> original_image_id
        if request.coco_data:
            for img in request.coco_data.get("images", []):
                filename = img.get("file_name", "")
                image_id_map[filename] = img.get("id")

        # If we have COCO data, get images from there
        if request.coco_data:
            for img in request.coco_data.get("images", []):
                filename = img.get("file_name", "")
                basename = Path(filename).name
                # Try matching by relative path first, then by basename
                if filename in image_lookup_rel:
                    all_image_paths.append(image_lookup_rel[filename])
                elif basename in image_lookup:
                    all_image_paths.append(image_lookup[basename])
                elif Path(filename).exists():
                    all_image_paths.append(filename)
        else:
            # Just use all images from directories
            all_image_paths = list(image_lookup.values())

        total_images = len(all_image_paths)

        if total_images == 0:
            return LabelingJobResponse(
                success=False,
                error="No images found in specified directories"
            )

        # Handle preview mode (limit to N images for testing)
        if request.preview_mode:
            preview_limit = min(request.preview_count, total_images)
            all_image_paths = all_image_paths[:preview_limit]
            total_images = preview_limit
            logger.info(f"[RELABEL] Preview mode enabled: processing first {total_images} images")
        else:
            # Warn about large datasets (may take hours/days to process)
            if total_images > 1000:
                logger.warning(f"[RELABEL] Large dataset detected: {total_images} images in mode '{request.relabel_mode}'. Processing may take several hours.")

        # Determine classes to label
        classes_to_label = request.new_classes if request.new_classes else []

        if not classes_to_label and request.coco_data:
            # Fall back to classes from existing dataset (for improve_segmentation,
            # or add/replace when user didn't specify new classes)
            classes_to_label = [c["name"] for c in request.coco_data.get("categories", [])]
            logger.info(f"[RELABEL] Using {len(classes_to_label)} classes from existing COCO data: {classes_to_label}")

        # Check for conflicting output directory with active jobs
        for existing_job in labeling_jobs.values():
            if (existing_job.get("status") in (JobStatus.QUEUED, JobStatus.PROCESSING)
                    and existing_job.get("output_dir") == request.output_dir):
                return LabelingJobResponse(
                    success=False,
                    error=f"A relabeling job is already running/queued with the same output directory: {request.output_dir}"
                )

        # Create job with unique output subdirectory
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_subdir = f"relabeling_{timestamp}_{job_id[:8]}"
        output_dir = Path(request.output_dir) / job_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[RELABEL] Job output directory: {output_dir}")

        labeling_jobs[job_id] = {
            "job_id": job_id,
            "job_type": "relabeling",
            "status": JobStatus.QUEUED,
            "total_images": total_images,
            "processed_images": 0,
            "total_objects_found": 0,
            "objects_by_class": {cls: 0 for cls in classes_to_label},
            "current_image": "",
            "output_dir": str(output_dir),
            "output_formats": request.output_formats,
            "errors": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            # Store request params
            "_image_paths": all_image_paths,
            "_image_lookup": image_lookup,
            "_image_id_map": image_id_map,  # filename -> original_image_id
            "_classes": classes_to_label,
            "_relabel_mode": request.relabel_mode,
            "_coco_data": request.coco_data,
            "_min_confidence": request.min_confidence,
            "_simplify_polygons": request.simplify_polygons,
            "_simplify_tolerance": getattr(request, "simplify_tolerance", 2.0),
            "_preview_mode": request.preview_mode,  # If True, this is a preview/test run
            "_deduplication_strategy": request.deduplication_strategy,  # Strategy for deduplication: confidence or area
        }

        # Persist relabeling job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="relabeling",
                    service="segmentation",
                    request_params={
                        "classes": classes_to_label,
                        "relabel_mode": request.relabel_mode,
                        "output_formats": request.output_formats,
                        "min_confidence": request.min_confidence,
                    },
                    total_items=total_images,
                    output_path=str(output_dir),
                )
                logger.info(f"Relabeling job {job_id} persisted to database")
            except Exception as e:
                logger.warning(f"Failed to persist relabeling job to database: {e}")

        # Start background task with concurrency control
        async def run_relabeling():
            async with labeling_job_semaphore:
                logger.info(f"[RELABEL] Job {job_id} acquired semaphore (max {MAX_CONCURRENT_LABELING_JOBS} concurrent)")
                labeling_jobs[job_id]["status"] = JobStatus.PROCESSING
                labeling_jobs[job_id]["started_at"] = datetime.now().isoformat()

                if state.db:
                    try:
                        state.db.update_job_status(job_id, "running", started_at=datetime.now())
                    except Exception as e:
                        logger.warning(f"Failed to update relabeling job status in database: {e}")

                try:
                    logger.info(f"[RELABEL] Calling _process_relabeling_job for {job_id}")
                    await _process_relabeling_job(job_id)
                    labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
                    logger.info(f"[RELABEL] Job {job_id} completed successfully")

                    # Update database with completion
                    if state.db:
                        try:
                            job = labeling_jobs[job_id]
                            state.db.complete_job(
                                job_id=job_id,
                                status="completed",
                                result_summary={
                                    "total_objects_found": job.get("total_objects_found", 0),
                                    "objects_by_class": job.get("objects_by_class", {}),
                                },
                                processing_time_ms=job.get("processing_time_ms", 0),
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update relabeling job completion in database: {e}")

                except Exception as e:
                    logger.exception(f"Relabeling job {job_id} failed: {e}")
                    labeling_jobs[job_id]["status"] = JobStatus.FAILED
                    labeling_jobs[job_id]["errors"].append(str(e))

                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id=job_id,
                                status="failed",
                                error_message=str(e),
                            )
                        except Exception as db_err:
                            logger.warning(f"Failed to update relabeling job failure in database: {db_err}")

                finally:
                    labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    logger.info(f"[RELABEL] Job {job_id} released semaphore")

        asyncio.create_task(run_relabeling())

        logger.info(f"Started relabeling job {job_id} - mode: {request.relabel_mode}, {total_images} images")

        return LabelingJobResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Relabeling job started. Mode: {request.relabel_mode}, {total_images} images. Output: {output_dir}",
            total_images=total_images
        )

    except Exception as e:
        logger.error(f"Failed to start relabeling job: {e}")
        return LabelingJobResponse(
            success=False,
            error=str(e)
        )


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


def _draw_annotations_on_image(
    image: np.ndarray,
    annotations: List[Dict],
    category_map: Dict[int, str],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True
) -> np.ndarray:
    """Draw annotations (bboxes, masks, labels) on an image for visualization.

    Args:
        image: BGR image array
        annotations: List of COCO-format annotations for this image
        category_map: Dict mapping category_id to category name
        draw_masks: Whether to draw segmentation masks
        draw_boxes: Whether to draw bounding boxes
        draw_labels: Whether to draw class labels

    Returns:
        Annotated image (BGR)
    """
    vis_image = image.copy()
    h, w = vis_image.shape[:2]

    # Generate distinct colors for each category
    np.random.seed(42)  # For consistent colors
    colors = {}
    for cat_id in category_map.keys():
        colors[cat_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))

    # Draw all masks with a single overlay operation (O(1) copies instead of O(N))
    if draw_masks:
        overlay = vis_image.copy()
        for ann in annotations:
            cat_id = ann.get("category_id", 1)
            color = colors.get(cat_id, (0, 255, 0))
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
        del overlay

    # Draw bounding boxes and labels
    for ann in annotations:
        cat_id = ann.get("category_id", 1)
        cat_name = category_map.get(cat_id, f"class_{cat_id}")
        color = colors.get(cat_id, (0, 255, 0))

        if draw_boxes and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)

        if draw_labels and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            label = f"{cat_name}"
            if "score" in ann:
                label += f" {ann['score']:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x, y - text_h - 4), (x + text_w + 4, y), color, -1)
            cv2.putText(vis_image, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def _save_preview_image(
    output_dir: Path,
    image: np.ndarray,
    image_name: str,
    annotations: List[Dict],
    category_map: Dict[int, str],
    preview_idx: int
) -> Optional[str]:
    """Save a preview image with annotations drawn.

    Returns the relative path to the saved preview, or None if failed.
    """
    try:
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        # Draw annotations
        vis_image = _draw_annotations_on_image(
            image, annotations, category_map,
            draw_masks=True, draw_boxes=True, draw_labels=True
        )

        # Save with a consistent naming scheme
        preview_name = f"preview_{preview_idx:04d}_{Path(image_name).stem}.jpg"
        preview_path = previews_dir / preview_name

        # Resize if too large (max 1024px on longest side) for faster loading
        h, w = vis_image.shape[:2]
        max_size = 1024
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            vis_image = cv2.resize(vis_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(preview_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(preview_path)
    except Exception as e:
        logger.warning(f"Failed to save preview image: {e}")
        return None


def _get_image_paths_from_output_dir(output_dir: Path) -> List[str]:
    """Reconstruct image paths from checkpoint/coco file in output directory.

    This is used when resuming a job after service restart.
    """
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Try to load from annotations.json or checkpoint
    coco_path = output_dir / "annotations.json"
    checkpoint_path = output_dir / "checkpoint.json"

    coco_data = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_data = checkpoint.get("coco_data")
        except Exception:
            pass

    if coco_data is None and coco_path.exists():
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
        except Exception:
            pass

    if coco_data and "images" in coco_data:
        # Get image directory from first image path or use output_dir parent
        images_dir = output_dir.parent / "images"
        if not images_dir.exists():
            images_dir = output_dir.parent

        for img_info in coco_data["images"]:
            file_name = img_info.get("file_name", "")
            # Try to find the image file
            for search_dir in [images_dir, output_dir.parent, output_dir]:
                potential_path = search_dir / file_name
                if potential_path.exists():
                    image_paths.append(str(potential_path))
                    break

    return image_paths


def _apply_padding_to_bbox(bbox: List[float], padding: int, img_width: int, img_height: int) -> List[float]:
    """Apply padding to a bounding box while keeping it within image bounds.

    Args:
        bbox: [x, y, width, height] format bounding box
        padding: Pixels of padding to add on each side
        img_width: Image width for bounds checking
        img_height: Image height for bounds checking

    Returns:
        Padded bbox [x, y, width, height] clamped to image bounds
    """
    if padding <= 0:
        return bbox

    x, y, w, h = bbox
    # Expand the bbox by padding on each side
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_width, x + w + padding)
    y2 = min(img_height, y + h + padding)

    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


async def _thermal_cooldown(cooldown_s: float, vram_monitor) -> None:
    """Sleep for cooldown_s, then add extra delay if GPU temperature is too high.

    Thresholds are read from environment variables:
      GPU_TEMP_THROTTLE_C  (default 80): start adding extra sleep
      GPU_TEMP_MAX_C       (default 87): max extra sleep (10 s)

    Extra sleep scales linearly from 2 s at throttle temp to 10 s at max temp.
    """
    await asyncio.sleep(cooldown_s)

    if vram_monitor is None:
        return

    temp = vram_monitor.get_temperature()
    if temp is None:
        return

    temp_throttle = int(os.environ.get("GPU_TEMP_THROTTLE_C", "80"))
    temp_max = int(os.environ.get("GPU_TEMP_MAX_C", "87"))

    if temp >= temp_throttle:
        ratio = min(1.0, (temp - temp_throttle) / max(1, temp_max - temp_throttle))
        extra = 2.0 + ratio * 8.0  # 2 s → 10 s
        logger.info(f"GPU temp {temp}°C (throttle={temp_throttle}°C) — extra cooldown {extra:.1f}s")
        await asyncio.sleep(extra)


async def _process_labeling_job(job_id: str, resume_from: int = 0):
    """Process a labeling job - detect and segment objects in images.

    Args:
        job_id: The job identifier
        resume_from: Image index to resume from (0 = start fresh)
    """
    import gc
    import torch
    from PIL import Image as PILImage

    job = labeling_jobs[job_id]
    start_time = time.time()

    image_paths = list(job["_image_paths"])  # local copy — OOM retries insert here, not into the job dict
    classes = job["_classes"]  # These are the search prompts
    class_mapping = job.get("_class_mapping")  # Maps prompts to final class names
    min_confidence = job["_min_confidence"]
    min_area = job["_min_area"]
    max_instances = job["_max_instances"]
    simplify_polygons = job["_simplify_polygons"]
    simplify_tolerance = job["_simplify_tolerance"]
    task_type = job["_task_type"]
    padding = job.get("_padding", 0)  # Pixels of padding around bboxes
    save_visualizations = job.get("_save_visualizations", True)  # Save preview images
    deduplication_strategy = job.get("_deduplication_strategy", "confidence")  # Deduplication strategy
    output_dir = Path(job["output_dir"])
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint_interval = 10  # Save checkpoint every N images
    max_previews = 50  # Maximum number of preview images to keep
    # Calculate dynamic preview interval based on total images (aim for ~30-50 previews)
    total_images = len(image_paths)
    preview_interval = max(1, total_images // max_previews) if total_images > max_previews else 1
    gc_interval = 20  # Run garbage collection every N images
    yield_interval = 1  # Yield to event loop every N images
    cooldown_ms = int(os.environ.get("LABELING_COOLDOWN_MS", "200"))
    cooldown_s = cooldown_ms / 1000.0
    # Cooldown between class inferences within a single image.
    # Prevents GPU from running at 100% continuously when there are many classes,
    # which causes overheating. Set to 0 to disable.
    inter_class_cooldown_ms = int(os.environ.get("INTER_CLASS_COOLDOWN_MS", "50"))
    inter_class_cooldown_s = inter_class_cooldown_ms / 1000.0

    # Initialize preview tracking
    preview_paths = job.get("_preview_paths", [])
    preview_count = len(preview_paths)
    images_with_detections = 0

    # Initialize quality metrics tracking
    all_scores = []  # Track all detection scores for avg_confidence
    low_confidence_count = 0  # Detections with score < 0.5
    images_without_detections = 0
    consecutive_errors = 0
    max_consecutive_errors = 10  # Abort if this many consecutive errors
    max_error_rate = 0.1  # Abort if error rate exceeds 10%

    # Initialize optimization modules
    prompt_optimizer = get_prompt_optimizer()
    detection_validator = get_detection_validator()

    # Initialize warnings list for class-level issues
    job["warnings"] = job.get("warnings", [])

    # Initialize VRAM monitor if available
    vram_monitor = None
    if VRAM_MONITOR_AVAILABLE and VRAMMonitor is not None:
        vram_monitor = VRAMMonitor(threshold=0.6, check_interval=1)
        logger.info(f"VRAMMonitor initialized for job {job_id}")

    # Determine final category names
    # If class_mapping exists, use unique mapped values as categories
    # Otherwise, use the original class names
    if class_mapping:
        # Get unique final class names (values from the mapping)
        # Preserve order by using the first appearance
        final_classes = []
        seen = set()
        for cls in classes:
            final_name = class_mapping.get(cls, cls)
            if final_name not in seen:
                final_classes.append(final_name)
                seen.add(final_name)
    else:
        final_classes = classes

    # Initialize or load COCO structure
    coco_result = None
    annotation_id = 1

    # Try to load checkpoint if resuming
    if resume_from > 0 and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_result = checkpoint.get("coco_data")
            annotation_id = checkpoint.get("next_annotation_id", 1)
            # Restore objects_by_class counts
            if "objects_by_class" in checkpoint:
                job["objects_by_class"] = checkpoint["objects_by_class"]
                job["total_objects_found"] = sum(checkpoint["objects_by_class"].values())
            logger.info(f"Resuming job {job_id} from image {resume_from}, annotation_id {annotation_id}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            coco_result = None

    # Initialize fresh if no checkpoint loaded
    if coco_result is None:
        coco_result = {
            "info": {
                "description": "Auto-labeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(final_classes)]
        }
        resume_from = 0  # Reset if no valid checkpoint

    # Expose coco_result reference for partial-annotations endpoint
    job["_coco_result"] = coco_result

    # Map final class names to category IDs
    category_map = {cls: i + 1 for i, cls in enumerate(final_classes)}

    # Process images starting from resume_from
    for img_idx, img_path in enumerate(image_paths):
        # Skip already processed images when resuming
        if img_idx < resume_from:
            continue

        # Check for cancellation
        if job.get("status") in (JobStatus.CANCELLED, "cancelled"):
            logger.info(f"Job {job_id} cancelled at image {img_idx}")
            _save_labeling_checkpoint(
                checkpoint_path, coco_result, annotation_id,
                img_idx, job["objects_by_class"]
            )
            return

        job["current_image"] = Path(img_path).name

        # Log progress every 10 images
        if img_idx == 0 or (img_idx + 1) % 10 == 0:
            logger.info(f"[LABEL] Job {job_id}: Processing image {img_idx + 1}/{len(image_paths)} ({100*(img_idx+1)/len(image_paths):.1f}%)")

        try:
            # Timeout per image to prevent indefinite blocking (30 seconds)
            async with asyncio.timeout(30.0):
                # Load image with validation
                image = cv2.imread(img_path)
                if image is None:
                    job["errors"].append(f"Failed to load: {img_path}")
                    consecutive_errors += 1
                    continue

                h, w = image.shape[:2]

                # Validate image dimensions (skip if too small or too large)
                if h < 10 or w < 10:
                    job["errors"].append(f"Image too small ({w}x{h}): {img_path}")
                    consecutive_errors += 1
                    continue
                if h > 8192 or w > 8192:
                    job["errors"].append(f"Image too large ({w}x{h}), may cause OOM: {img_path}")
                    consecutive_errors += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(image_rgb)

                # Add image to COCO
                image_id = img_idx + 1
                coco_result["images"].append({
                    "id": image_id,
                    "file_name": Path(img_path).name,
                    "width": w,
                    "height": h
                })

                # Track annotations added for this image (for deduplication)
                image_annotations_start_idx = len(coco_result["annotations"])

                # Process each class (cls is the search prompt)
                for cls in classes:
                    # Get final class name (may be different if mapping exists)
                    final_class = class_mapping.get(cls, cls) if class_mapping else cls

                    # Get optimized prompt for better detection
                    optimized_prompt = prompt_optimizer.get_primary_prompt(cls)

                    # Run SAM3 text prompt segmentation using optimized prompt
                    inputs = state.sam3_processor(
                        images=pil_image,
                        text=optimized_prompt,  # Use optimized prompt for detection
                        return_tensors="pt"
                    ).to(state.device)

                    with torch.no_grad():
                        outputs = state.sam3_model(**inputs)

                    # Post-process results
                    target_sizes = inputs.get("original_sizes")
                    if target_sizes is not None:
                        target_sizes = target_sizes.tolist()
                    else:
                        target_sizes = [(h, w)]

                    results = state.sam3_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=min_confidence,
                        mask_threshold=min_confidence,
                        target_sizes=target_sizes
                    )[0]
                    del inputs, outputs  # Free GPU tensors immediately after inference

                    if 'masks' not in results or len(results['masks']) == 0:
                        del results
                        continue

                    # Process each detected instance
                    instances_added = 0
                    for mask, score in zip(results['masks'], results['scores']):
                        if instances_added >= max_instances:
                            break

                        score_val = score.cpu().item()
                        if score_val < min_confidence:
                            continue

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                        # Calculate area
                        area = int(np.sum(mask_np > 0))
                        if area < min_area:
                            continue

                        # Get bounding box
                        contours, _ = cv2.findContours(
                            (mask_np * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        if not contours:
                            continue

                        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
                        bbox = [float(x), float(y), float(bw), float(bh)]

                        # Validate detection using DetectionValidator
                        is_valid, rejection_reason, adjusted_score = detection_validator.validate_detection(
                            mask=mask_np,
                            bbox=bbox,
                            class_name=final_class,
                            image_size=(w, h),
                            score=score_val,
                        )

                        if not is_valid:
                            logger.debug(f"Rejected detection for '{final_class}': {rejection_reason}")
                            continue

                        # Track quality metrics
                        all_scores.append(adjusted_score)
                        if adjusted_score < 0.5:
                            low_confidence_count += 1

                        # Apply padding if configured
                        if padding > 0:
                            bbox = _apply_padding_to_bbox(bbox, padding, w, h)

                        # Build annotation (use final_class for category lookup)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[final_class],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "_score": adjusted_score,  # Track score for potential dedup
                        }

                        # Add segmentation if task requires it
                        if task_type in ["segmentation", "both"]:
                            polygons = state.object_extractor.mask_to_polygon(
                                (mask_np * 255).astype(np.uint8),
                                simplify=simplify_polygons,
                                tolerance=simplify_tolerance
                            )
                            if polygons:
                                annotation["segmentation"] = polygons

                        coco_result["annotations"].append(annotation)
                        annotation_id += 1
                        instances_added += 1

                        # Track by final class name
                        job["objects_by_class"][final_class] = job["objects_by_class"].get(final_class, 0) + 1
                        job["total_objects_found"] += 1

                    del results  # Free results after processing all instances

                    # Inter-class cooldown: let the GPU thermal load drop between inferences.
                    # Synchronize first to ensure GPU is idle during the sleep.
                    if inter_class_cooldown_s > 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        await asyncio.sleep(inter_class_cooldown_s)

            # Deduplicate annotations for this image (remove overlapping detections)
            image_annotations = coco_result["annotations"][image_annotations_start_idx:]
            if len(image_annotations) > 1:
                deduped = deduplicate_annotations(image_annotations, iou_threshold=0.5, strategy=deduplication_strategy)
                removed_count = len(image_annotations) - len(deduped)
                if removed_count > 0:
                    # Update the annotations list
                    coco_result["annotations"] = coco_result["annotations"][:image_annotations_start_idx] + deduped
                    # Adjust counts
                    job["total_objects_found"] -= removed_count
                    # Re-assign annotation IDs for this image's annotations
                    for i, ann in enumerate(coco_result["annotations"][image_annotations_start_idx:]):
                        ann["id"] = image_annotations_start_idx + i + 1
                    annotation_id = len(coco_result["annotations"]) + 1
                    logger.debug(f"Deduplicated {removed_count} overlapping annotations for {Path(img_path).name}")

            # Track images without detections
            current_image_detections = len(coco_result["annotations"]) - image_annotations_start_idx
            if current_image_detections == 0:
                images_without_detections += 1

            # Reset consecutive error counter on successful processing
            consecutive_errors = 0

            # Save preview image if this image had detections
            if save_visualizations and current_image_detections > 0:
                images_with_detections += 1
                # Save preview periodically (every N images with detections, up to max_previews)
                if images_with_detections % preview_interval == 0 and preview_count < max_previews:
                    # Get annotations for this image
                    image_annotations = [
                        ann for ann in coco_result["annotations"]
                        if ann.get("image_id") == image_id
                    ]
                    # Create reverse category map (id -> name)
                    category_id_to_name = {i + 1: cls for i, cls in enumerate(final_classes)}

                    preview_path = _save_preview_image(
                        output_dir, image, Path(img_path).name,
                        image_annotations, category_id_to_name, preview_count
                    )
                    if preview_path:
                        preview_paths.append(preview_path)
                        preview_count += 1
                        job["_preview_paths"] = preview_paths
                        logger.debug(f"Saved preview {preview_count}/{max_previews} at image {img_idx + 1}")

            # Update progress after successful processing
            job["processed_images"] = img_idx + 1

            # Save checkpoint periodically and update database
            if (img_idx + 1) % checkpoint_interval == 0:
                _save_labeling_checkpoint(
                    checkpoint_path, coco_result, annotation_id,
                    img_idx + 1, job["objects_by_class"]
                )
                logger.debug(f"Saved checkpoint at image {img_idx + 1}")

                # Clear retry count on successful processing
                if "_retry_count" in job and img_path in job["_retry_count"]:
                    del job["_retry_count"][img_path]

                # Update quality metrics in real-time (so frontend can display them)
                job["quality_metrics"] = {
                    "avg_confidence": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                    "images_with_detections": images_with_detections,
                    "images_without_detections": images_without_detections,
                    "low_confidence_count": low_confidence_count,
                    "total_detections": len(all_scores),
                }

                # Update progress in database
                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=img_idx + 1,
                            current_item=Path(img_path).name,
                            progress_details={
                                "total_objects_found": job["total_objects_found"],
                                "objects_by_class": job["objects_by_class"],
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update progress in database: {e}")

            # Garbage collection - use VRAMMonitor if available, otherwise periodic
            should_cleanup = False
            if vram_monitor is not None:
                should_cleanup = vram_monitor.should_cleanup()
            else:
                should_cleanup = (img_idx + 1) % gc_interval == 0

            if should_cleanup:
                # Clear image references
                del image, image_rgb, pil_image
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                if vram_monitor is not None:
                    stats = vram_monitor.get_vram_stats()
                    logger.debug(f"VRAM cleanup at image {img_idx + 1}: {stats.get('allocated_gb', 0):.2f}GB")

            # Cooldown between images: fixed delay + adaptive extra if GPU runs hot
            await _thermal_cooldown(cooldown_s, vram_monitor)

        except asyncio.TimeoutError:
            # Image processing took too long (>30s), skip it
            consecutive_errors += 1
            error_msg = f"Timeout processing {img_path} (>30s), skipping"
            job["errors"].append(error_msg)
            logger.warning(f"Job {job_id}: {error_msg}")
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            # OOM: Clean memory and retry with exponential backoff
            consecutive_errors += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            retry_count = job.get("_retry_count", {}).get(img_path, 0)
            if retry_count < 3:
                # Set up retry
                job.setdefault("_retry_count", {})[img_path] = retry_count + 1
                wait_time = 2 ** retry_count  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"OOM on {img_path}, retry {retry_count + 1}/3 after {wait_time}s")
                await asyncio.sleep(wait_time)
                # Re-add to process queue (will be picked up next iteration)
                image_paths.insert(img_idx + 1, img_path)
                continue
            else:
                job["errors"].append(f"OOM error after 3 retries: {img_path}")
                logger.error(f"OOM error after 3 retries: {img_path}")

        except Exception as e:
            consecutive_errors += 1
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            logger.error(f"Error processing {img_path}: {e}")
            # Save checkpoint on error so we can resume
            _save_labeling_checkpoint(
                checkpoint_path, coco_result, annotation_id,
                img_idx, job["objects_by_class"]
            )
            # Clean up on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Check abort conditions
        processed = img_idx + 1
        error_count = len(job.get("errors", []))

        # Abort if too many consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            job["status"] = "failed"
            job["error"] = f"Too many consecutive errors ({consecutive_errors}). Check input data or GPU memory."
            logger.error(f"Job {job_id} aborted: {job['error']}")
            break

        # Abort if error rate exceeds threshold (after processing enough images)
        if processed >= 50 and error_count / processed > max_error_rate:
            job["status"] = "failed"
            job["error"] = f"Error rate too high: {error_count}/{processed} ({100*error_count/processed:.1f}%). Check class names or image quality."
            logger.error(f"Job {job_id} aborted: {job['error']}")
            break

        # Add warning if a class has no detections after significant processing
        if processed >= 20 and processed % 20 == 0:
            for cls in classes:
                final_cls = class_mapping.get(cls, cls) if class_mapping else cls
                if job["objects_by_class"].get(final_cls, 0) == 0:
                    warning = f"No detections for '{cls}' after {processed} images"
                    if warning not in job.get("warnings", []):
                        job.setdefault("warnings", []).append(warning)
                        logger.warning(f"Job {job_id}: {warning}")

    # Update quality metrics
    job["quality_metrics"] = {
        "avg_confidence": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "images_with_detections": images_with_detections,
        "images_without_detections": images_without_detections,
        "low_confidence_count": low_confidence_count,
        "total_detections": len(all_scores),
    }

    # Check if job was aborted
    if job.get("status") == "failed":
        # Save partial results even on failure
        coco_path = output_dir / "annotations_partial.json"
        with open(coco_path, 'w') as f:
            json.dump(coco_result, f, indent=2)
        logger.info(f"Saved partial results to {coco_path}")
        return

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    # Save COCO JSON
    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    # Save in other formats if requested
    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug(f"Removed checkpoint file after successful completion")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")

    logger.info(f"Labeling job {job_id} completed: {job['total_objects_found']} objects found")


def _save_labeling_checkpoint(checkpoint_path: Path, coco_data: Dict, next_annotation_id: int,
                               last_processed_idx: int, objects_by_class: Dict[str, int]):
    """Save checkpoint for labeling job recovery."""
    checkpoint = {
        "coco_data": coco_data,
        "next_annotation_id": next_annotation_id,
        "last_processed_idx": last_processed_idx,
        "objects_by_class": objects_by_class,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


async def _process_relabeling_job(job_id: str):
    """Process a relabeling job."""
    import gc
    import torch
    from PIL import Image as PILImage

    logger.info(f"[RELABEL] _process_relabeling_job started for {job_id}")

    job = labeling_jobs[job_id]
    start_time = time.time()

    relabel_mode = job["_relabel_mode"]
    logger.info(f"[RELABEL] Mode: {relabel_mode}, Classes: {job.get('_classes', [])[:5]}...")
    coco_data = job.get("_coco_data")
    image_paths = list(job["_image_paths"])  # local copy — OOM retries insert here, not into the job dict
    image_lookup = job["_image_lookup"]
    image_id_map = job.get("_image_id_map", {})  # filename -> original_image_id
    classes = job["_classes"]
    min_confidence = job["_min_confidence"]
    simplify_polygons = job["_simplify_polygons"]
    deduplication_strategy = job.get("_deduplication_strategy", "confidence")  # Deduplication strategy
    output_dir = Path(job["output_dir"])
    gc_interval = 20  # Run garbage collection every N images
    cooldown_ms = int(os.environ.get("LABELING_COOLDOWN_MS", "200"))
    cooldown_s = cooldown_ms / 1000.0
    inter_class_cooldown_ms = int(os.environ.get("INTER_CLASS_COOLDOWN_MS", "50"))
    inter_class_cooldown_s = inter_class_cooldown_ms / 1000.0

    vram_monitor = None
    if VRAM_MONITOR_AVAILABLE and VRAMMonitor is not None:
        vram_monitor = VRAMMonitor(threshold=0.6, check_interval=1)

    # Initialize result based on mode
    if relabel_mode == "add" and coco_data:
        # Start with existing data
        coco_result = coco_data.copy()
        # Add new categories
        existing_cats = {c["name"] for c in coco_result.get("categories", [])}
        max_cat_id = max([c["id"] for c in coco_result.get("categories", [])], default=0)
        for cls in classes:
            if cls not in existing_cats:
                max_cat_id += 1
                coco_result["categories"].append({"id": max_cat_id, "name": cls, "supercategory": ""})
        annotation_id = max([a["id"] for a in coco_result.get("annotations", [])], default=0) + 1
    else:
        # Start fresh
        coco_result = {
            "info": {
                "description": "Relabeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(classes)]
        }
        annotation_id = 1

    # Expose coco_result reference for partial-annotations endpoint
    job["_coco_result"] = coco_result

    category_map = {c["name"]: c["id"] for c in coco_result.get("categories", [])}
    prompt_optimizer = get_prompt_optimizer()
    logger.info(f"[RELABEL] Category map: {category_map}")
    logger.info(f"[RELABEL] Starting to process {len(image_paths)} images")

    # Process images
    for img_idx, img_path in enumerate(image_paths):
        # Check for cancellation
        if job.get("status") in (JobStatus.CANCELLED, "cancelled"):
            logger.info(f"[RELABEL] Job {job_id} cancelled at image {img_idx}")
            return

        job["current_image"] = Path(img_path).name

        if img_idx == 0 or (img_idx + 1) % 10 == 0:
            logger.info(f"[RELABEL] Job {job_id}: Processing image {img_idx + 1}/{len(image_paths)} ({100*(img_idx+1)/len(image_paths):.1f}%) - {Path(img_path).name}")

        try:
            # Timeout per image to prevent indefinite blocking (30 seconds)
            async with asyncio.timeout(30.0):
                image = cv2.imread(img_path)
                if image is None:
                    continue

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # Use original image ID if available, otherwise generate sequential ID
            filename = Path(img_path).name
            image_id = image_id_map.get(filename, img_idx + 1)

            # Add image if not already in coco_result
            existing_image_ids = {img["id"] for img in coco_result.get("images", [])}
            if image_id not in existing_image_ids:
                coco_result["images"].append({
                    "id": image_id,
                    "file_name": filename,
                    "width": w,
                    "height": h
                })

            # Track annotations added for this image (for deduplication)
            image_annotations_start_idx = len(coco_result["annotations"])

            # Handle improve_segmentation mode differently
            if relabel_mode == "improve_segmentation" and coco_data:
                # Get existing annotations for this image (bbox-only annotations)
                img_anns = [a for a in coco_data.get("annotations", [])
                           if a.get("image_id") == image_id and not a.get("segmentation")]

                for ann in img_anns:
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        # Use SAM3 to generate segmentation from bbox
                        x, y, bw, bh = bbox
                        input_box = [[x, y, x + bw, y + bh]]

                        inputs = state.sam3_processor(
                            images=pil_image,
                            input_boxes=[input_box],
                            return_tensors="pt"
                        ).to(state.device)

                        with torch.no_grad():
                            outputs = state.sam3_model(**inputs)

                        masks = state.sam3_processor.post_process_masks(
                            outputs.pred_masks,
                            inputs["original_sizes"],
                            inputs["reshaped_input_sizes"]
                        )
                        del inputs, outputs  # Free GPU tensors immediately after inference

                        if len(masks) > 0 and len(masks[0]) > 0:
                            mask = masks[0][0]
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                            del masks  # Free after extracting the numpy array

                            if mask_np.shape != (h, w):
                                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                            polygons = state.object_extractor.mask_to_polygon(
                                mask_np,
                                simplify=simplify_polygons,
                                tolerance=2.0
                            )

                            if polygons:
                                # Update annotation with segmentation
                                ann["segmentation"] = polygons
                                ann["area"] = int(np.sum(mask_np > 0))
                                job["total_objects_found"] += 1
                        else:
                            del masks  # Free even if no usable masks

            else:
                # Get existing annotations for this image (for deduplication in "add" mode)
                existing_anns_for_image = []
                if relabel_mode == "add" and coco_data:
                    existing_anns_for_image = [a for a in coco_data.get("annotations", [])
                                               if a.get("image_id") == image_id]

                # Regular labeling with text prompts
                for cls in classes:
                    if cls not in category_map:
                        continue

                    optimized_prompt = prompt_optimizer.get_primary_prompt(cls)
                    inputs = state.sam3_processor(
                        images=pil_image,
                        text=optimized_prompt,
                        return_tensors="pt"
                    ).to(state.device)

                    with torch.no_grad():
                        outputs = state.sam3_model(**inputs)

                    target_sizes = [(h, w)]
                    results = state.sam3_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=min_confidence,
                        mask_threshold=min_confidence,
                        target_sizes=target_sizes
                    )[0]
                    del inputs, outputs  # Free GPU tensors immediately after inference

                    if 'masks' not in results:
                        del results
                        continue

                    for mask, score in zip(results['masks'], results['scores']):
                        score_val = score.cpu().item()
                        if score_val < min_confidence:
                            continue

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                        area = int(np.sum(mask_np > 0))
                        if area < 100:
                            continue

                        contours, _ = cv2.findContours(
                            (mask_np * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        if not contours:
                            continue

                        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))

                        polygons = state.object_extractor.mask_to_polygon(
                            (mask_np * 255).astype(np.uint8),
                            simplify=simplify_polygons,
                            tolerance=2.0
                        )

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[cls],
                            "bbox": [float(x), float(y), float(bw), float(bh)],
                            "area": area,
                            "iscrowd": 0,
                            "_score": score_val,
                        }

                        if polygons:
                            annotation["segmentation"] = polygons

                        # Check for duplicates in "add" mode (IoU threshold 0.5)
                        is_duplicate = False
                        if relabel_mode == "add" and existing_anns_for_image:
                            new_bbox = annotation["bbox"]
                            for exist_ann in existing_anns_for_image:
                                exist_bbox = exist_ann.get("bbox", [])
                                if len(exist_bbox) == 4:
                                    # Calculate IoU
                                    iou = _calculate_iou(new_bbox, exist_bbox)
                                    if iou > 0.5:
                                        is_duplicate = True
                                        break

                        if not is_duplicate:
                            coco_result["annotations"].append(annotation)
                            annotation_id += 1

                            job["objects_by_class"][cls] = job["objects_by_class"].get(cls, 0) + 1
                            job["total_objects_found"] += 1

                    del results  # Free results after processing all instances

                    # Inter-class cooldown: let the GPU thermal load drop between inferences.
                    if inter_class_cooldown_s > 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        await asyncio.sleep(inter_class_cooldown_s)

            # Deduplicate annotations for this image (remove overlapping detections across all classes)
            image_anns = coco_result["annotations"][image_annotations_start_idx:]
            if len(image_anns) > 1:
                deduped = deduplicate_annotations(image_anns, iou_threshold=0.5, strategy=deduplication_strategy)
                removed_count = len(image_anns) - len(deduped)
                if removed_count > 0:
                    coco_result["annotations"] = coco_result["annotations"][:image_annotations_start_idx] + deduped
                    job["total_objects_found"] -= removed_count
                    for i, ann in enumerate(coco_result["annotations"][image_annotations_start_idx:]):
                        ann["id"] = image_annotations_start_idx + i + 1
                    annotation_id = len(coco_result["annotations"]) + 1
                    logger.debug(f"[RELABEL] Deduplicated {removed_count} overlapping annotations for {Path(img_path).name}")

            # Update progress after successful processing
            job["processed_images"] = img_idx + 1

            # Garbage collection
            if (img_idx + 1) % gc_interval == 0:
                del image, image_rgb, pil_image
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

            # Cooldown between images: fixed delay + adaptive extra if GPU runs hot
            await _thermal_cooldown(cooldown_s, vram_monitor)

        except asyncio.TimeoutError:
            # Image processing took too long (>30s), skip it
            error_msg = f"[RELABEL] Timeout processing {img_path} (>30s), skipping"
            job["errors"].append(error_msg)
            logger.warning(error_msg)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        except Exception as e:
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    logger.info(f"Relabeling job {job_id} completed")


def _export_to_yolo(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to YOLO format."""
    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)

    labels_dir = yolo_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Build image lookup
    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco_data.get("images", [])}
    image_names = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

    # Category mapping (YOLO uses 0-indexed)
    categories = coco_data.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write labels
    for img_id, anns in anns_by_image.items():
        if img_id not in image_dims:
            continue

        w, h = image_dims[img_id]
        filename = Path(image_names[img_id]).stem + ".txt"

        lines = []
        for ann in anns:
            cat_idx = cat_map.get(ann["category_id"], 0)
            bbox = ann.get("bbox", [0, 0, 0, 0])

            # Convert to YOLO format (center x, center y, width, height) normalized
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(labels_dir / filename, 'w') as f:
            f.write('\n'.join(lines))

    # Write classes.txt
    with open(yolo_dir / "classes.txt", 'w') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")


def _export_to_voc(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to Pascal VOC format."""
    voc_dir = output_dir / "voc"
    voc_dir.mkdir(exist_ok=True)

    annotations_dir = voc_dir / "Annotations"
    annotations_dir.mkdir(exist_ok=True)

    # Build lookups
    image_info = {img["id"]: img for img in coco_data.get("images", [])}
    cat_names = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write XML files
    for img_id, anns in anns_by_image.items():
        if img_id not in image_info:
            continue

        img = image_info[img_id]
        filename = Path(img["file_name"]).stem + ".xml"

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<annotation>',
            f'  <filename>{img["file_name"]}</filename>',
            '  <size>',
            f'    <width>{img["width"]}</width>',
            f'    <height>{img["height"]}</height>',
            '    <depth>3</depth>',
            '  </size>',
        ]

        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            cat_name = cat_names.get(ann["category_id"], "unknown")

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])

            xml_lines.extend([
                '  <object>',
                f'    <name>{cat_name}</name>',
                '    <bndbox>',
                f'      <xmin>{xmin}</xmin>',
                f'      <ymin>{ymin}</ymin>',
                f'      <xmax>{xmax}</xmax>',
                f'      <ymax>{ymax}</ymax>',
                '    </bndbox>',
                '  </object>',
            ])

        xml_lines.append('</annotation>')

        with open(annotations_dir / filename, 'w') as f:
            f.write('\n'.join(xml_lines))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
