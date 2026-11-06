"""
Object extraction endpoints.
"""

import os
import json
import time
import logging
import asyncio
import uuid
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import cv2
import numpy as np

from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import (
    AnalyzeDatasetRequest, AnalyzeDatasetResponse, CategoryInfo,
    ExtractObjectsRequest, ExtractObjectsResponse,
    ExtractCustomObjectsRequest, ExtractCustomObjectsResponse,
    ExtractionJobStatus,
    ExtractSingleObjectRequest, ExtractSingleObjectResponse,
    JobStatus, AnnotationType, ExtractionMethod,
)
from app.service_state import state, extraction_jobs

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract/analyze-dataset", response_model=AnalyzeDatasetResponse, tags=["Object Extraction"])
async def analyze_dataset_for_extraction(request: AnalyzeDatasetRequest):
    """
    Analyze a COCO dataset to determine annotation types.

    Returns counts of annotations with segmentation vs bbox-only,
    and a recommendation for extraction method.
    """
    try:
        # Get COCO data
        coco_data = None
        if request.coco_data:
            coco_data = request.coco_data
        elif request.coco_json_path:
            json_path = Path(request.coco_json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"COCO JSON not found: {request.coco_json_path}")
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
        else:
            raise ValueError("Either coco_data or coco_json_path must be provided")

        # Analyze dataset
        analysis = state.object_extractor.analyze_dataset(coco_data)

        # Convert categories to CategoryInfo
        categories = [
            CategoryInfo(
                id=cat["id"],
                name=cat["name"],
                count=cat["count"],
                with_segmentation=cat["with_segmentation"],
                bbox_only=cat["bbox_only"]
            )
            for cat in analysis["categories"]
        ]

        return AnalyzeDatasetResponse(
            success=True,
            total_images=analysis["total_images"],
            total_annotations=analysis["total_annotations"],
            annotations_with_segmentation=analysis["annotations_with_segmentation"],
            annotations_bbox_only=analysis["annotations_bbox_only"],
            categories=categories,
            recommendation=analysis["recommendation"],
            sample_annotation=analysis["sample_annotation"]
        )

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return AnalyzeDatasetResponse(
            success=False,
            error=str(e)
        )


@router.post("/extract/objects", response_model=ExtractObjectsResponse, tags=["Object Extraction"])
async def extract_objects(request: ExtractObjectsRequest):
    """
    Extract objects from a COCO dataset as transparent PNG images.

    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    try:
        # Get COCO data
        coco_data = None
        source_path = ""
        if request.coco_data:
            coco_data = request.coco_data
            source_path = "uploaded_data"
            logger.info(f"Received COCO data with {len(coco_data.get('annotations', []))} annotations")
        elif request.coco_json_path:
            json_path = Path(request.coco_json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"COCO JSON not found: {request.coco_json_path}")
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            source_path = request.coco_json_path
        else:
            raise ValueError("Either coco_data or coco_json_path must be provided")

        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count total objects to extract
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        if request.categories_to_extract:
            valid_cat_ids = {cid for cid, name in categories.items() if name in request.categories_to_extract}
        else:
            valid_cat_ids = set(categories.keys())

        total_objects = sum(
            1 for ann in coco_data.get("annotations", [])
            if ann.get("category_id") in valid_cat_ids
            and ann.get("bbox", [0, 0, 0, 0])[2] * ann.get("bbox", [0, 0, 0, 0])[3] >= request.min_object_area
        )

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": total_objects,
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {},
            "output_dir": request.output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None,
            "duplicates_prevented": 0,
            "deduplication_enabled": request.deduplication.enabled if request.deduplication else True
        }

        # Persist extraction job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="extraction",
                    service="segmentation",
                    request_params={
                        "source_path": source_path,
                        "images_dir": request.images_dir,
                        "categories_to_extract": request.categories_to_extract,
                        "use_sam3_for_bbox": request.use_sam3_for_bbox,
                        "force_bbox_only": request.force_bbox_only,
                    },
                    total_items=total_objects,
                    output_path=request.output_dir,
                )
                state.db.update_job_status(job_id, "running", started_at=datetime.now())
            except Exception as e:
                logger.warning(f"Failed to persist extraction job to DB: {e}")

        # Define extraction task
        async def run_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # This callback runs from a thread pool, but dict updates are atomic in CPython
                extracted = progress["extracted"]
                failed = progress["failed"]
                current_category = progress.get("current_category", "")
                by_category = progress.get("by_category", {})
                extraction_jobs[job_id]["extracted_objects"] = extracted
                extraction_jobs[job_id]["failed_objects"] = failed
                extraction_jobs[job_id]["current_category"] = current_category
                extraction_jobs[job_id]["categories_progress"] = by_category

                # Save checkpoint file so progress survives a partial run
                try:
                    checkpoint_path = os.path.join(request.output_dir, "extraction_progress.json")
                    with open(checkpoint_path, "w") as _cp_f:
                        json.dump(
                            {
                                "extracted": extracted,
                                "failed": failed,
                                "completed_categories": list(by_category.keys()),
                                "current_category": current_category,
                                "timestamp": datetime.now().isoformat(),
                            },
                            _cp_f,
                            indent=2,
                        )
                except Exception:
                    pass

                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=extracted,
                            failed_items=failed,
                            current_item=current_category,
                            progress_details={"by_category": by_category},
                        )
                    except Exception:
                        pass

            try:
                result = await state.object_extractor.extract_from_dataset(
                    coco_data=coco_data,
                    images_dir=str(images_dir),
                    output_dir=request.output_dir,
                    categories_to_extract=request.categories_to_extract or None,
                    use_sam3_for_bbox=request.use_sam3_for_bbox,
                    force_bbox_only=request.force_bbox_only,
                    force_sam3_resegmentation=request.force_sam3_resegmentation,
                    force_sam3_text_prompt=request.force_sam3_text_prompt,
                    padding=request.padding,
                    min_object_area=request.min_object_area,
                    save_individual_coco=request.save_individual_coco,
                    progress_callback=progress_callback,
                    deduplication_config=request.deduplication
                )
                extraction_jobs[job_id]["extracted_objects"] = result["extracted"]
                extraction_jobs[job_id]["failed_objects"] = result["failed"]
                extraction_jobs[job_id]["categories_progress"] = result["by_category"]
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["extracted_files"] = result.get("extracted_files", [])[:1000]
                extraction_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000
                extraction_jobs[job_id]["duplicates_prevented"] = result.get("deduplication_stats", {}).get("duplicates_prevented", 0)
                extraction_jobs[job_id]["status"] = JobStatus.COMPLETED

                if state.db:
                    try:
                        state.db.complete_job(
                            job_id,
                            "completed",
                            result_summary={
                                "extracted": result["extracted"],
                                "failed": result["failed"],
                                "by_category": result["by_category"],
                                "duplicates_prevented": result.get("deduplication_stats", {}).get("duplicates_prevented", 0),
                            },
                        )
                    except Exception:
                        pass

            except Exception as e:
                # Use logger.exception to get full traceback for debugging
                logger.exception(f"Extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

                if state.db:
                    try:
                        state.db.complete_job(job_id, "failed", error_message=str(e))
                    except Exception:
                        pass

            finally:
                # Always set completed_at, even if job failed
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background using asyncio.create_task for proper async execution
        asyncio.create_task(run_extraction())

        dedup_status = "enabled (IoU=0.7)" if (request.deduplication and request.deduplication.enabled) or request.deduplication is None else "disabled"
        logger.info(f"Started extraction job {job_id} with {total_objects} objects (deduplication: {dedup_status})")

        return ExtractObjectsResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Extraction job queued. {total_objects} objects to extract. Deduplication: {dedup_status}"
        )

    except Exception as e:
        logger.error(f"Failed to start extraction: {e}")
        return ExtractObjectsResponse(
            success=False,
            error=str(e)
        )


@router.post("/extract/custom-objects", response_model=ExtractCustomObjectsResponse, tags=["Object Extraction"])
async def extract_custom_objects(request: ExtractCustomObjectsRequest):
    """
    Extract custom objects using text prompts (no COCO JSON required).

    This endpoint allows you to specify object names directly and segment them
    from images using SAM3 text prompt mode.

    Process:
    1. Scans all images in images_dir
    2. For each object name in the list, runs SAM3 text prompt segmentation
    3. Extracts all detected instances as transparent PNGs
    4. Organizes results by object type: output_dir/{object_name}/

    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    try:
        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Validate object names
        if not request.object_names or len(request.object_names) == 0:
            raise ValueError("At least one object name must be provided")

        # Create job
        job_id = str(uuid.uuid4())

        # Count images for progress tracking
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        total_images = sum(
            1 for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        )

        if total_images == 0:
            raise ValueError(f"No images found in {request.images_dir}")

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": 0,  # Unknown until extraction
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {name: 0 for name in request.object_names},
            "output_dir": request.output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None,
            "duplicates_prevented": 0,
            "deduplication_enabled": request.deduplication.enabled if request.deduplication else True,
            "total_images": total_images,
            "current_image": ""
        }

        # Persist custom extraction job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="extraction",
                    service="segmentation",
                    request_params={
                        "images_dir": request.images_dir,
                        "object_names": request.object_names,
                        "total_images": total_images,
                        "extraction_type": "custom",
                    },
                    total_items=total_images,
                    output_path=request.output_dir,
                )
                state.db.update_job_status(job_id, "running", started_at=datetime.now())
            except Exception as e:
                logger.warning(f"Failed to persist custom extraction job to DB: {e}")

        # Define extraction task
        async def run_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # Update job status with progress info
                extracted = progress.get("extracted", 0)
                failed = progress.get("failed", 0)
                current_image = progress.get("current_image", "")
                extraction_jobs[job_id]["extracted_objects"] = extracted
                extraction_jobs[job_id]["failed_objects"] = failed
                extraction_jobs[job_id]["duplicates_prevented"] = progress.get("duplicates_prevented", 0)
                extraction_jobs[job_id]["current_image"] = current_image

                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=extracted,
                            failed_items=failed,
                            current_item=current_image,
                        )
                    except Exception:
                        pass

            try:
                result = await state.object_extractor.extract_custom_objects(
                    images_dir=str(images_dir),
                    output_dir=request.output_dir,
                    object_names=request.object_names,
                    padding=request.padding,
                    min_object_area=request.min_object_area,
                    save_individual_coco=request.save_individual_coco,
                    deduplication_config=request.deduplication.dict() if request.deduplication else None,
                    progress_callback=progress_callback
                )

                # Update final job status
                extraction_jobs[job_id]["total_objects"] = result.get("total_objects_extracted", 0)
                extraction_jobs[job_id]["extracted_objects"] = result.get("total_objects_extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = result.get("failed_extractions", 0)
                extraction_jobs[job_id]["categories_progress"] = result.get("by_category", {})
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000
                extraction_jobs[job_id]["duplicates_prevented"] = result.get("duplicates_prevented", 0)

                if result.get("success"):
                    extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                    logger.info(f"Custom extraction job {job_id} completed: {result['total_objects_extracted']} objects")
                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id,
                                "completed",
                                result_summary={
                                    "total_objects_extracted": result.get("total_objects_extracted", 0),
                                    "failed_extractions": result.get("failed_extractions", 0),
                                    "by_category": result.get("by_category", {}),
                                },
                            )
                        except Exception:
                            pass
                else:
                    extraction_jobs[job_id]["status"] = JobStatus.FAILED
                    logger.error(f"Custom extraction job {job_id} failed")
                    if state.db:
                        try:
                            state.db.complete_job(job_id, "failed", error_message="Extraction returned failure")
                        except Exception:
                            pass

            except Exception as e:
                logger.exception(f"Custom extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

                if state.db:
                    try:
                        state.db.complete_job(job_id, "failed", error_message=str(e))
                    except Exception:
                        pass

            finally:
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background
        asyncio.create_task(run_extraction())

        dedup_status = "enabled (IoU=0.7)" if (request.deduplication and request.deduplication.enabled) or request.deduplication is None else "disabled"
        logger.info(
            f"Started custom extraction job {job_id} for {len(request.object_names)} object types "
            f"across {total_images} images (deduplication: {dedup_status})"
        )

        return ExtractCustomObjectsResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Custom extraction job queued. Will search for {len(request.object_names)} object types in {total_images} images. Deduplication: {dedup_status}"
        )

    except Exception as e:
        logger.error(f"Failed to start custom extraction: {e}")
        return ExtractCustomObjectsResponse(
            success=False,
            error=str(e)
        )


def _db_row_to_extraction_job(db_job: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB job row to the extraction job dict format used by the list endpoint."""
    result_summary = db_job.get("result_summary") or {}
    total = db_job.get("total_items", 0)
    extracted = db_job.get("processed_items", 0)
    failed = db_job.get("failed_items", 0)
    progress = round(((extracted + failed) / total * 100), 1) if total > 0 else 0.0

    return {
        "job_id": db_job.get("id", ""),
        "type": "extraction",
        "job_type": "extraction",
        "status": db_job.get("status", "unknown"),
        "progress": progress,
        "created_at": db_job.get("created_at", datetime.now().isoformat()),
        "total_objects": result_summary.get("extracted", 0) + result_summary.get("failed", 0) if result_summary else total,
        "extracted_objects": result_summary.get("extracted", extracted),
        "failed_objects": result_summary.get("failed", failed),
        "current_category": db_job.get("current_item", ""),
        "output_dir": db_job.get("output_path", ""),
        "started_at": db_job.get("started_at"),
        "completed_at": db_job.get("completed_at"),
        "processing_time_ms": db_job.get("processing_time_ms", 0),
    }


@router.get("/extract/jobs", tags=["Object Extraction"])
async def list_extraction_jobs():
    """List all extraction jobs (active from memory, historical from database)."""
    jobs = []
    seen_ids: set = set()

    # First: active jobs from memory (most detailed real-time data)
    for job_id, job in extraction_jobs.items():
        # Convert JobStatus enum to string for JSON serialization
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Calculate progress
        total = job.get("total_objects", 0)
        extracted = job.get("extracted_objects", 0)
        failed = job.get("failed_objects", 0)
        progress = round(((extracted + failed) / total * 100), 1) if total > 0 else 0.0

        jobs.append({
            "job_id": job_id,
            "type": "extraction",  # Frontend expects 'type' field
            "job_type": "extraction",
            "status": status_str,
            "progress": progress,  # Add progress percentage
            "created_at": job.get("created_at", job.get("started_at", datetime.now().isoformat())),
            "total_objects": job.get("total_objects", 0),
            "extracted_objects": job.get("extracted_objects", 0),
            "failed_objects": job.get("failed_objects", 0),
            "current_category": job.get("current_category", ""),
            "output_dir": job.get("output_dir", ""),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
        })
        seen_ids.add(job_id)

    # Second: historical/completed jobs from DB not present in memory
    if state.db:
        try:
            db_jobs = state.db.list_jobs(service="segmentation", job_type="extraction", limit=50)
            for db_job in db_jobs:
                job_id = db_job.get("id", "")
                if job_id and job_id not in seen_ids:
                    jobs.append(_db_row_to_extraction_job(db_job))
        except Exception:
            pass

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/extract/jobs/{job_id}", response_model=ExtractionJobStatus, tags=["Object Extraction"])
async def get_extraction_job_status(job_id: str):
    """Get the status of an extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = extraction_jobs[job_id]

    # Calculate progress percentage
    total = job.get("total_objects", 0)
    extracted = job.get("extracted_objects", 0)
    failed = job.get("failed_objects", 0)
    progress = ((extracted + failed) / total * 100) if total > 0 else 0.0

    return ExtractionJobStatus(
        **{k: v for k, v in job.items() if k != "progress"},
        progress=round(progress, 1)
    )


@router.post("/extract/single-object", response_model=ExtractSingleObjectResponse, tags=["Object Extraction"])
async def extract_single_object(request: ExtractSingleObjectRequest):
    """
    Extract a single object for preview.

    Returns the extracted object as base64-encoded PNG with transparency.
    """
    start_time = time.time()

    try:
        # Load image
        image = None
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif request.image_path:
            image_path = Path(request.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {request.image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Failed to load image")

        # Extract object
        result = await state.object_extractor.extract_single_object(
            image=image,
            annotation=request.annotation,
            category_name=request.category_name,
            use_sam3=request.use_sam3,
            padding=request.padding,
            force_bbox_only=request.force_bbox_only,
            force_sam3_resegmentation=request.force_sam3_resegmentation,
            force_sam3_text_prompt=request.force_sam3_text_prompt
        )

        processing_time = (time.time() - start_time) * 1000

        if not result["success"]:
            return ExtractSingleObjectResponse(
                success=False,
                processing_time_ms=processing_time,
                error=result.get("error", "Unknown error")
            )

        return ExtractSingleObjectResponse(
            success=True,
            cropped_image_base64=result["cropped_image_base64"],
            mask_base64=result.get("mask_base64"),
            annotation_type=AnnotationType(result["annotation_type"]),
            method_used=ExtractionMethod(result["method_used"]),
            original_bbox=result["original_bbox"],
            extracted_size=result["extracted_size"],
            mask_coverage=result["mask_coverage"],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Single object extraction failed: {e}")
        return ExtractSingleObjectResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@router.post("/extract/imagenet", tags=["Object Extraction"])
async def extract_from_imagenet(
    root_dir: str,
    output_dir: str,
    padding: int = 5,
    min_object_area: int = 100,
    max_objects_per_class: Optional[int] = None
):
    """
    Extract objects from ImageNet-style directory structure.

    Expected structure:
        root_dir/
        ├── class1/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── class2/
        │   └── ...

    Uses SAM3 with class name as text prompt for segmentation.
    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    if not state.sam3_available:
        return {
            "success": False,
            "error": "SAM3 is required for ImageNet extraction but not available"
        }

    try:
        # Validate root directory
        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count classes
        try:
            classes = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
            num_classes = len(classes)
        except Exception as e:
            raise ValueError(f"Failed to read root directory: {e}")

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": 0,  # Will be updated as we discover images
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {},
            "output_dir": output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None,
            "extraction_type": "imagenet"
        }

        # Persist ImageNet extraction job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="extraction",
                    service="segmentation",
                    request_params={
                        "root_dir": root_dir,
                        "num_classes": num_classes,
                        "extraction_type": "imagenet",
                    },
                    total_items=num_classes,
                    output_path=output_dir,
                )
                state.db.update_job_status(job_id, "running", started_at=datetime.now())
            except Exception as e:
                logger.warning(f"Failed to persist ImageNet extraction job to DB: {e}")

        # Define extraction task
        async def run_imagenet_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # Update job progress
                current_class = progress.get("current_class", "")
                extracted = progress.get("extracted", 0)
                failed = progress.get("failed", 0)
                extraction_jobs[job_id]["current_category"] = current_class
                extraction_jobs[job_id]["extracted_objects"] = extracted
                extraction_jobs[job_id]["failed_objects"] = failed

                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=extracted,
                            failed_items=failed,
                            current_item=current_class,
                        )
                    except Exception:
                        pass

            try:
                start_time = time.time()
                result = await state.object_extractor.extract_from_imagenet_structure(
                    root_dir=root_dir,
                    output_dir=output_dir,
                    padding=padding,
                    min_object_area=min_object_area,
                    max_objects_per_class=max_objects_per_class,
                    progress_callback=progress_callback
                )

                if result.get("success"):
                    extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id,
                                "completed",
                                result_summary={
                                    "total_extracted": result.get("total_extracted", 0),
                                    "total_failed": result.get("total_failed", 0),
                                    "classes": result.get("classes", {}),
                                },
                            )
                        except Exception:
                            pass
                else:
                    extraction_jobs[job_id]["status"] = JobStatus.FAILED
                    if state.db:
                        try:
                            state.db.complete_job(job_id, "failed", error_message="ImageNet extraction returned failure")
                        except Exception:
                            pass

                extraction_jobs[job_id]["total_objects"] = result.get("total_extracted", 0) + result.get("total_failed", 0)
                extraction_jobs[job_id]["extracted_objects"] = result.get("total_extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = result.get("total_failed", 0)
                extraction_jobs[job_id]["categories_progress"] = result.get("classes", {})
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["processing_time_ms"] = (time.time() - start_time) * 1000

            except Exception as e:
                logger.exception(f"ImageNet extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

                if state.db:
                    try:
                        state.db.complete_job(job_id, "failed", error_message=str(e))
                    except Exception:
                        pass

            finally:
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background
        asyncio.create_task(run_imagenet_extraction())
        logger.info(f"Started ImageNet extraction job {job_id} from {root_dir} ({num_classes} classes)")

        return {
            "success": True,
            "job_id": job_id,
            "status": "pending",
            "message": f"ImageNet extraction job queued. Processing {num_classes} classes."
        }

    except Exception as e:
        logger.error(f"Failed to start ImageNet extraction: {e}")
        return {
            "success": False,
            "error": str(e)
        }

