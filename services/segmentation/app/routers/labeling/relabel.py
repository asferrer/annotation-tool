"""
Start endpoint for relabeling jobs.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from app.models.extraction_schemas import StartRelabelingRequest, LabelingJobResponse, JobStatus
from app.service_state import state, labeling_jobs, wait_for_sam3, labeling_job_semaphore, MAX_CONCURRENT_LABELING_JOBS
from app.routers.labeling.relabel_processing import _process_relabeling_job

logger = logging.getLogger(__name__)

router = APIRouter()


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
    if not state.sam3_available:
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
            "image_directories": list(request.image_directories),
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
                        "image_directories": list(request.image_directories),
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
