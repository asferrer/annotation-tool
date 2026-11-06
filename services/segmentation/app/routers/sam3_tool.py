"""
SAM3 tool endpoints: image segmentation and dataset conversion.
"""

import json
import time
import logging
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import base64
import cv2
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.extraction_schemas import (
    SAM3SegmentImageRequest, SAM3SegmentImageResponse,
    SAM3ConvertDatasetRequest, SAM3ConvertDatasetResponse,
    SAM3ConversionJobStatus, JobStatus, AnnotationType,
)
from app.service_state import (
    state, sam3_conversion_jobs, wait_for_sam3,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_image_path(image_path: str) -> Path:
    """Resolve image path, searching subdirectories if the direct path doesn't exist."""
    p = Path(image_path)
    if p.exists():
        return p
    # Search recursively in the parent directory for a file with the same name
    parent = p.parent
    name = p.name
    for candidate in parent.rglob(name):
        return candidate
    raise FileNotFoundError(f"Image not found: {image_path}")


@router.post("/sam3/segment-image", response_model=SAM3SegmentImageResponse, tags=["SAM3 Tool"])
async def sam3_segment_image(request: SAM3SegmentImageRequest):
    """
    Segment an object in an image using SAM3 with box or point prompt.

    Returns the segmentation mask and polygon coordinates.
    """
    start_time = time.time()

    # Wait for SAM3 if still loading
    if not state.sam3_available:
        logger.info("Waiting for SAM3 to load for image segmentation...")
        await wait_for_sam3(timeout=60.0)

    if not state.sam3_available:
        error_msg = state.sam3_load_error or "SAM3 not available"
        return SAM3SegmentImageResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=error_msg
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Load image
        image = None
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.image_path:
            image_path = _resolve_image_path(request.image_path)
            image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Failed to load image")

        h, w = image.shape[:2]

        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Prepare inputs based on prompt type.
        # use_tracker_postprocess=True when Sam3TrackerProcessor was used (no reshaped_input_sizes).
        inputs = None
        use_tracker_postprocess = False
        if request.bbox:
            x, y, bw, bh = request.bbox
            input_box = [[x, y, x + bw, y + bh]]  # xywh → xyxy
            if request.text_prompt:
                # Combined text+bbox mode: better quality when the category is known
                inputs = state.sam3_processor(
                    images=pil_image,
                    text=request.text_prompt,
                    input_boxes=[input_box],
                    return_tensors="pt"
                ).to(state.device)
            else:
                inputs = state.sam3_processor(
                    images=pil_image,
                    input_boxes=[input_box],
                    return_tensors="pt"
                ).to(state.device)
        elif request.point:
            # Sam3Processor does not accept input_points — use Sam3TrackerProcessor instead.
            # Sam3Model still requires input_ids, so we inject them from Sam3Processor.
            if state.sam3_tracker_processor is None:
                raise ValueError("Sam3TrackerProcessor not available for point segmentation")
            # Sam3TrackerProcessor expects 4-level nesting: [image][object][N_points][2]
            input_points_fmt = [[[[float(request.point[0]), float(request.point[1])]]]]
            input_labels_fmt = [[[1]]]  # [image][object][N_points]
            tracker_inputs = state.sam3_tracker_processor(
                images=pil_image,
                input_points=input_points_fmt,
                input_labels=input_labels_fmt,
                return_tensors="pt"
            ).to(state.device)
            text_inputs_pt = state.sam3_processor(
                images=pil_image, text="object", return_tensors="pt"
            ).to(state.device)
            allowed = {"pixel_values", "input_points", "input_labels", "original_sizes"}
            inputs = {k: v for k, v in tracker_inputs.items() if k in allowed}
            inputs["input_ids"] = text_inputs_pt["input_ids"]
            if "attention_mask" in text_inputs_pt:
                inputs["attention_mask"] = text_inputs_pt["attention_mask"]
            use_tracker_postprocess = True
        elif request.text_prompt:
            inputs = state.sam3_processor(
                images=pil_image,
                text=request.text_prompt,
                return_tensors="pt"
            ).to(state.device)
        else:
            raise ValueError("Must provide bbox, point, or text_prompt")

        # Cast floating inputs to model dtype (prevents float/half mismatch when model is fp16)
        model_dtype = next(state.sam3_model.parameters()).dtype
        if model_dtype != torch.float32:
            for key in list(inputs.keys()):
                val = inputs[key]
                if hasattr(val, 'is_floating_point') and val.is_floating_point():
                    inputs[key] = val.to(model_dtype)

        # Run SAM3
        with torch.no_grad():
            outputs = state.sam3_model(**inputs)

        # Post-process:
        #   text-only → post_process_instance_segmentation (PCS mode)
        #   point     → Sam3TrackerProcessor.post_process_masks (2 args, no reshaped_input_sizes)
        #   bbox/bbox+text → Sam3Processor.post_process_masks (3 args)
        text_only = not request.bbox and not request.point and bool(request.text_prompt)

        if text_only:
            # PCS scores are typically very low (<0.05) even for valid matches.
            # Use threshold=0.0 to retain all proposals, pick the one with highest score.
            results = state.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                mask_threshold=0.5,
                target_sizes=[[h, w]]
            )[0]
            if len(results["masks"]) == 0:
                return SAM3SegmentImageResponse(
                    success=False,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error="No mask generated"
                )
            best_idx = int(results["scores"].argmax().item())
            mask_np = results["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
            confidence = float(results["scores"][best_idx].item())
            del results
        elif use_tracker_postprocess:
            # Point mode: Sam3Model PCS output → select instance covering the click point.
            results_pt = state.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                mask_threshold=0.0,
                target_sizes=[[h, w]]
            )[0]
            if len(results_pt["masks"]) == 0:
                return SAM3SegmentImageResponse(
                    success=False,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error="No mask generated"
                )
            click_pt = request.point
            covering_pt = []
            for i, mask_t in enumerate(results_pt["masks"]):
                m = mask_t.cpu().numpy()
                r, c = int(click_pt[1]), int(click_pt[0])
                if 0 <= r < h and 0 <= c < w and m[r, c]:
                    covering_pt.append((i, int(mask_t.float().sum().item())))
            if covering_pt:
                covering_pt.sort(key=lambda x: x[1])  # prefer smallest (most precise)
                best_idx = covering_pt[0][0]
            else:
                best_idx = int(results_pt["scores"].argmax().item())
            confidence = float(results_pt["scores"][best_idx].item())
            mask_np = results_pt["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
            del results_pt
        else:
            # Bbox mode: Sam3Processor lacks post_process_masks — use post_process_instance_segmentation.
            # Select the proposal with highest IoU against the user-drawn bbox.
            results_bb = state.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                mask_threshold=0.5,
                target_sizes=[[h, w]]
            )[0]
            if len(results_bb["masks"]) == 0:
                return SAM3SegmentImageResponse(
                    success=False,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error="No mask generated"
                )
            # Compute overlap of each proposal mask with the input bbox region.
            bx, by, bw_, bh_ = request.bbox
            bx2, by2 = int(bx + bw_), int(by + bh_)
            bx, by = int(bx), int(by)
            bbox_area = max(1, (bx2 - bx) * (by2 - by))
            best_idx, best_overlap = 0, -1.0
            for i, mask_t in enumerate(results_bb["masks"]):
                m = mask_t.cpu().numpy()
                crop = m[max(0, by):min(h, by2), max(0, bx):min(w, bx2)]
                overlap = float(crop.sum()) / bbox_area
                if overlap > best_overlap:
                    best_overlap, best_idx = overlap, i
            confidence = float(results_bb["scores"][best_idx].item())
            mask_np = results_bb["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
            # Clip mask to the user-drawn bbox — zero out everything outside it
            clip = np.zeros_like(mask_np)
            clip[max(0, by):min(h, by2), max(0, bx):min(w, bx2)] = 255
            mask_np = cv2.bitwise_and(mask_np, clip)
            del results_bb

        del inputs, outputs

        # Ensure correct size
        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # Calculate bbox and area
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
            bbox = [float(x), float(y), float(bw), float(bh)]
        else:
            bbox = [0.0, 0.0, float(w), float(h)]

        area = int(np.sum(mask_np > 0))

        # Convert to polygon if requested
        segmentation_polygon = None
        segmentation_coco = None
        if request.return_polygon:
            polygons = state.object_extractor.mask_to_polygon(
                mask_np,
                simplify=request.simplify_polygon,
                tolerance=request.simplify_tolerance
            )
            if polygons:
                segmentation_coco = polygons
                # Convert to [[x,y], [x,y], ...] format
                segmentation_polygon = [
                    [[polygons[0][i], polygons[0][i+1]] for i in range(0, len(polygons[0]), 2)]
                ]

        # Encode mask if requested
        mask_base64 = None
        if request.return_mask:
            success, encoded = cv2.imencode('.png', mask_np)
            if success:
                mask_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

        return SAM3SegmentImageResponse(
            success=True,
            mask_base64=mask_base64,
            segmentation_polygon=segmentation_polygon[0] if segmentation_polygon else None,
            segmentation_coco=segmentation_coco,
            bbox=bbox,
            area=area,
            confidence=confidence,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        logger.error(f"SAM3 segmentation failed: {e}")
        return SAM3SegmentImageResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@router.post("/sam3/convert-dataset", response_model=SAM3ConvertDatasetResponse, tags=["SAM3 Tool"])
async def sam3_convert_dataset(request: SAM3ConvertDatasetRequest):
    """
    Convert bbox-only annotations to segmentations using SAM3.

    Runs asynchronously. Use GET /sam3/jobs/{job_id} to track progress.
    """
    if not state.sam3_available:
        return SAM3ConvertDatasetResponse(
            success=False,
            error="SAM3 not available"
        )

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

        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count annotations to convert
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        if request.categories_to_convert:
            valid_cat_ids = {cid for cid, name in categories.items() if name in request.categories_to_convert}
        else:
            valid_cat_ids = set(categories.keys())

        total_annotations = 0
        for ann in coco_data.get("annotations", []):
            if ann.get("category_id") not in valid_cat_ids:
                continue
            ann_type = state.object_extractor.detect_annotation_type(ann)
            if ann_type == AnnotationType.BBOX_ONLY or request.overwrite_existing:
                total_annotations += 1

        sam3_conversion_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_annotations": total_annotations,
            "converted_annotations": 0,
            "skipped_annotations": 0,
            "failed_annotations": 0,
            "current_image": "",
            "categories_progress": {},
            "output_path": request.output_path,
            "errors": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None
        }

        # Persist SAM3 conversion job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="sam3_conversion",
                    service="segmentation",
                    request_params={
                        "images_dir": request.images_dir,
                        "categories_to_convert": request.categories_to_convert,
                        "overwrite_existing": request.overwrite_existing,
                    },
                    total_items=total_annotations,
                    output_path=request.output_path,
                )
                state.db.update_job_status(job_id, "running", started_at=datetime.now())
            except Exception as e:
                logger.warning(f"Failed to persist SAM3 conversion job to DB: {e}")

        # Define conversion task
        async def run_conversion():
            sam3_conversion_jobs[job_id]["status"] = JobStatus.PROCESSING
            sam3_conversion_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # This callback runs from a thread pool, but dict updates are atomic in CPython
                converted = progress["converted"]
                skipped = progress["skipped"]
                failed = progress["failed"]
                current_image = progress.get("current_image", "")
                sam3_conversion_jobs[job_id]["converted_annotations"] = converted
                sam3_conversion_jobs[job_id]["skipped_annotations"] = skipped
                sam3_conversion_jobs[job_id]["failed_annotations"] = failed
                sam3_conversion_jobs[job_id]["current_image"] = current_image
                sam3_conversion_jobs[job_id]["categories_progress"] = progress.get("by_category", {})

                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=converted + skipped,
                            failed_items=failed,
                            current_item=current_image,
                            progress_details={"by_category": progress.get("by_category", {})},
                        )
                    except Exception:
                        pass

            try:
                result = await state.object_extractor.convert_bbox_to_segmentation(
                    coco_data=coco_data,
                    images_dir=str(images_dir),
                    output_path=request.output_path,
                    categories_to_convert=request.categories_to_convert or None,
                    overwrite_existing=request.overwrite_existing,
                    simplify_polygons=request.simplify_polygons,
                    simplify_tolerance=request.simplify_tolerance,
                    progress_callback=progress_callback,
                )

                if result.get("success"):
                    sam3_conversion_jobs[job_id]["status"] = JobStatus.COMPLETED
                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id,
                                "completed",
                                result_summary={
                                    "converted": result.get("converted", 0),
                                    "skipped": result.get("skipped", 0),
                                    "failed": result.get("failed", 0),
                                    "by_category": result.get("by_category", {}),
                                },
                            )
                        except Exception:
                            pass
                else:
                    sam3_conversion_jobs[job_id]["status"] = JobStatus.FAILED
                    if state.db:
                        try:
                            state.db.complete_job(job_id, "failed", error_message="Conversion returned failure")
                        except Exception:
                            pass

                sam3_conversion_jobs[job_id]["converted_annotations"] = result.get("converted", 0)
                sam3_conversion_jobs[job_id]["skipped_annotations"] = result.get("skipped", 0)
                sam3_conversion_jobs[job_id]["failed_annotations"] = result.get("failed", 0)
                sam3_conversion_jobs[job_id]["categories_progress"] = result.get("by_category", {})
                sam3_conversion_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                sam3_conversion_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000

            except Exception as e:
                # Use logger.exception to get full traceback for debugging
                logger.exception(f"Conversion job {job_id} failed: {e}")
                sam3_conversion_jobs[job_id]["status"] = JobStatus.FAILED
                sam3_conversion_jobs[job_id]["errors"].append(str(e))

                if state.db:
                    try:
                        state.db.complete_job(job_id, "failed", error_message=str(e))
                    except Exception:
                        pass

            finally:
                # Always set completed_at, even if job failed
                sam3_conversion_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background using asyncio.create_task for proper async execution
        asyncio.create_task(run_conversion())
        logger.info(f"Started SAM3 conversion job {job_id} with {total_annotations} annotations")

        return SAM3ConvertDatasetResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Conversion job queued. {total_annotations} annotations to convert."
        )

    except Exception as e:
        logger.error(f"Failed to start conversion: {e}")
        return SAM3ConvertDatasetResponse(
            success=False,
            error=str(e)
        )


def _db_row_to_sam3_conversion_job(db_job: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB job row to the SAM3 conversion job dict format."""
    result_summary = db_job.get("result_summary") or {}
    total = db_job.get("total_items", 0)
    converted = result_summary.get("converted", db_job.get("processed_items", 0))
    skipped = result_summary.get("skipped", 0)
    failed = result_summary.get("failed", db_job.get("failed_items", 0))
    progress = round(((converted + skipped + failed) / total * 100), 1) if total > 0 else 0.0

    return {
        "job_id": db_job.get("id", ""),
        "type": "sam3_conversion",
        "job_type": "sam3_conversion",
        "status": db_job.get("status", "unknown"),
        "progress": progress,
        "created_at": db_job.get("created_at", datetime.now().isoformat()),
        "total_annotations": total,
        "converted_annotations": converted,
        "skipped_annotations": skipped,
        "failed_annotations": failed,
        "current_image": db_job.get("current_item", ""),
        "output_path": db_job.get("output_path", ""),
        "started_at": db_job.get("started_at"),
        "completed_at": db_job.get("completed_at"),
        "processing_time_ms": db_job.get("processing_time_ms", 0),
    }


@router.get("/sam3/jobs", tags=["SAM3 Tool"])
async def list_sam3_conversion_jobs():
    """List all SAM3 conversion jobs (active from memory, historical from database)."""
    jobs = []
    seen_ids: set = set()

    # First: active jobs from memory (most detailed real-time data)
    for job_id, job in sam3_conversion_jobs.items():
        # Convert JobStatus enum to string for JSON serialization
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Calculate progress
        total = job.get("total_annotations", 0)
        converted = job.get("converted_annotations", 0)
        skipped = job.get("skipped_annotations", 0)
        failed = job.get("failed_annotations", 0)
        progress = round(((converted + skipped + failed) / total * 100), 1) if total > 0 else 0.0

        jobs.append({
            "job_id": job_id,
            "type": "sam3_conversion",  # Frontend expects 'type' field
            "job_type": "sam3_conversion",
            "status": status_str,
            "progress": progress,  # Add progress percentage
            "created_at": job.get("created_at", job.get("started_at", datetime.now().isoformat())),
            "total_annotations": job.get("total_annotations", 0),
            "converted_annotations": job.get("converted_annotations", 0),
            "skipped_annotations": job.get("skipped_annotations", 0),
            "failed_annotations": job.get("failed_annotations", 0),
            "current_image": job.get("current_image", ""),
            "output_path": job.get("output_path", ""),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
        })
        seen_ids.add(job_id)

    # Second: historical/completed jobs from DB not present in memory
    if state.db:
        try:
            db_jobs = state.db.list_jobs(service="segmentation", job_type="sam3_conversion", limit=50)
            for db_job in db_jobs:
                job_id = db_job.get("id", "")
                if job_id and job_id not in seen_ids:
                    jobs.append(_db_row_to_sam3_conversion_job(db_job))
        except Exception:
            pass

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/sam3/jobs/{job_id}", response_model=SAM3ConversionJobStatus, tags=["SAM3 Tool"])
async def get_sam3_conversion_job_status(job_id: str):
    """Get the status of a SAM3 conversion job."""
    if job_id not in sam3_conversion_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = sam3_conversion_jobs[job_id]

    # Calculate progress percentage
    total = job.get("total_annotations", 0)
    converted = job.get("converted_annotations", 0)
    skipped = job.get("skipped_annotations", 0)
    failed = job.get("failed_annotations", 0)
    progress = ((converted + skipped + failed) / total * 100) if total > 0 else 0.0

    return SAM3ConversionJobStatus(
        **{k: v for k, v in job.items() if k != "progress"},
        progress=round(progress, 1)
    )


# ---------------------------------------------------------------------------
# Endpoint: interactive point-based segmentation (Sam3TrackerModel)
# ---------------------------------------------------------------------------

class SegmentPointRequest(BaseModel):
    image_path: str
    points: List[List[float]]       # [[x1,y1], [x2,y2], ...]
    labels: List[int]               # 1=foreground, 0=background
    text_hint: Optional[str] = None  # category name for better PCS segmentation quality
    return_polygon: bool = True
    simplify_polygon: bool = True
    simplify_tolerance: float = 2.0


@router.post("/sam3/segment-point", tags=["SAM3 Tool"])
async def sam3_segment_point(request: SegmentPointRequest):
    """
    Segment an object using point clicks (Sam3TrackerProcessor + Sam3Model hybrid).

    Sam3TrackerProcessor handles input_points/input_labels preprocessing.
    Sam3Model performs inference (requires input_ids, injected via Sam3Processor).
    Click foreground (label=1) and background (label=0) points.
    """
    start_time = time.time()

    if not state.sam3_available:
        await wait_for_sam3(timeout=60.0)

    if not state.sam3_available:
        return {
            "success": False,
            "segmentation_coco": None,
            "bbox": None,
            "confidence": 0.0,
            "error": state.sam3_load_error or "SAM3 not available",
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    if state.sam3_tracker_processor is None:
        return {
            "success": False,
            "segmentation_coco": None,
            "bbox": None,
            "confidence": 0.0,
            "error": "Sam3TrackerProcessor not available",
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    try:
        import torch
        from PIL import Image as PILImage

        image_path = _resolve_image_path(request.image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Failed to load image")

        h, w = image.shape[:2]
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        fg_points = [p for p, lbl in zip(request.points, request.labels) if lbl == 1]
        if not fg_points:
            return {
                "success": False,
                "segmentation_coco": None,
                "bbox": None,
                "confidence": 0.0,
                "error": "No foreground points provided",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # Sam3TrackerProcessor expects 4-level nesting:
        # input_points: [image][object][N_points][2] — all points in one "object" query
        # input_labels: [image][object][N_points]
        input_points_fmt = [[[[float(p[0]), float(p[1])] for p in request.points]]]
        input_labels_fmt = [[[int(lbl) for lbl in request.labels]]]

        tracker_inputs = state.sam3_tracker_processor(
            images=pil_image,
            input_points=input_points_fmt,
            input_labels=input_labels_fmt,
            return_tensors="pt",
        ).to(state.device)

        # Sam3Model requires input_ids/text_embeds — inject from Sam3Processor.
        # Use text_hint (category name) if provided for higher segmentation quality;
        # otherwise fall back to generic "object" text.
        text_prompt = request.text_hint.strip() if request.text_hint else "object"
        text_inputs = state.sam3_processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(state.device)

        # Build combined inputs: point tensors from tracker, text tokens from sam3_processor.
        allowed_keys = {
            "pixel_values", "input_ids", "attention_mask", "pixel_mask",
            "text_embeds", "input_points", "input_labels", "original_sizes",
        }
        combined = {k: v for k, v in tracker_inputs.items() if k in allowed_keys}
        combined["input_ids"] = text_inputs["input_ids"]
        if "attention_mask" in text_inputs:
            combined["attention_mask"] = text_inputs["attention_mask"]

        model_dtype = next(state.sam3_model.parameters()).dtype
        if model_dtype != torch.float32:
            for key in list(combined.keys()):
                val = combined[key]
                if hasattr(val, 'is_floating_point') and val.is_floating_point():
                    combined[key] = val.to(model_dtype)

        with torch.no_grad():
            outputs = state.sam3_model(**combined)

        # Sam3Model runs in PCS mode → post_process_instance_segmentation returns
        # up to 200 mask proposals. Use threshold=0 to keep all proposals (PCS scores
        # can be low for generic text), mask_threshold=0.5 for clean binary masks.
        # Then select the proposal whose mask covers the foreground click point(s).
        results = state.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=0.5,
            target_sizes=[[h, w]]
        )[0]
        del combined, outputs

        if len(results["masks"]) == 0:
            return {
                "success": False,
                "segmentation_coco": None,
                "bbox": None,
                "confidence": 0.0,
                "error": "No mask generated",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # Find masks that cover the foreground click points, pick the most precise one.
        covering = []
        for i, mask_t in enumerate(results["masks"]):
            m = mask_t.cpu().numpy()
            n_covered = sum(
                1 for p in fg_points
                if 0 <= int(p[1]) < h and 0 <= int(p[0]) < w and m[int(p[1]), int(p[0])]
            )
            if n_covered > 0:
                area = int(mask_t.float().sum().item())
                covering.append((i, n_covered, area))

        if covering:
            # Sort by coverage desc, then area asc (more coverage, smaller area = more precise)
            covering.sort(key=lambda x: (-x[1], x[2]))
            best_idx = covering[0][0]
        else:
            best_idx = int(results["scores"].argmax().item())

        confidence = float(results["scores"][best_idx].item())
        mask_np = results["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
        del results

        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            bx, by, bw, bh = cv2.boundingRect(np.concatenate(contours))
            bbox = [float(bx), float(by), float(bw), float(bh)]
        else:
            bbox = None

        segmentation_coco = None
        if request.return_polygon:
            polygons = state.object_extractor.mask_to_polygon(
                mask_np,
                simplify=request.simplify_polygon,
                tolerance=request.simplify_tolerance,
            )
            if polygons:
                segmentation_coco = polygons

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return {
            "success": segmentation_coco is not None,
            "segmentation_coco": segmentation_coco,
            "bbox": bbox,
            "confidence": confidence,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    except Exception as e:
        logger.error(f"segment-point failed: {e}", exc_info=True)
        return {
            "success": False,
            "segmentation_coco": None,
            "bbox": None,
            "confidence": 0.0,
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000,
        }


