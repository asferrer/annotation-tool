"""
Annotations Router - COCO annotation CRUD operations.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Annotations"], prefix="/annotations")

# Per-path async locks prevent concurrent read-modify-write races on the same file
_coco_locks: Dict[str, asyncio.Lock] = {}


def _get_coco_lock(path: str) -> asyncio.Lock:
    if path not in _coco_locks:
        _coco_locks[path] = asyncio.Lock()
    return _coco_locks[path]

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


# ---------- Schemas ----------

class LoadAnnotationsRequest(BaseModel):
    coco_json_path: str


class SaveAnnotationsRequest(BaseModel):
    coco_json_path: str
    data: Dict[str, Any]


class CreateAnnotationRequest(BaseModel):
    coco_json_path: str
    image_id: int
    category_id: int
    bbox: List[float]
    segmentation: Optional[List[List[float]]] = None
    area: Optional[float] = None


class UpdateAnnotationRequest(BaseModel):
    coco_json_path: str
    annotation_id: int
    bbox: Optional[List[float]] = None
    category_id: Optional[int] = None
    segmentation: Optional[List[List[float]]] = None


class DeleteAnnotationRequest(BaseModel):
    coco_json_path: str
    annotation_id: int


class ReannotateRequest(BaseModel):
    coco_json_path: str
    image_id: int
    annotation_id: int


class SegmentBboxRequest(BaseModel):
    image_path: str
    bbox: List[float]  # [x, y, w, h] in image coordinates
    text_hint: Optional[str] = None  # category name to improve segmentation


class SegmentPointRequest(BaseModel):
    image_path: str
    points: List[List[float]]
    labels: List[int]
    text_hint: Optional[str] = None  # category name for better PCS segmentation quality
    return_polygon: bool = True
    simplify_polygon: bool = True
    simplify_tolerance: float = 2.0


# ---------- Helpers ----------

def _read_coco_sync(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_coco_sync(path: str, data: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        backup = p.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(p, backup)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def _read_coco(path: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_read_coco_sync, path)


async def _write_coco(path: str, data: Dict[str, Any]):
    await asyncio.to_thread(_write_coco_sync, path, data)


# ---------- Endpoints ----------

@router.post("/load")
async def load_annotations(request: LoadAnnotationsRequest):
    """Load a COCO JSON file and return its contents."""
    data = await _read_coco(request.coco_json_path)
    return {
        "images": data.get("images", []),
        "annotations": data.get("annotations", []),
        "categories": data.get("categories", []),
        "info": data.get("info", {}),
    }


@router.get("/images")
async def list_images(coco_json_path: str = Query(...)):
    """List all images in the dataset with annotation counts."""
    data = await _read_coco(coco_json_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_counts: Dict[int, int] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        ann_counts[img_id] = ann_counts.get(img_id, 0) + 1

    return [
        {**img, "annotation_count": ann_counts.get(img["id"], 0)}
        for img in images
    ]


@router.get("/images/{image_id}")
async def get_image_annotations(image_id: int, coco_json_path: str = Query(...)):
    """Get all annotations for a specific image."""
    data = await _read_coco(coco_json_path)
    annotations = [
        a for a in data.get("annotations", []) if a.get("image_id") == image_id
    ]
    image = next((i for i in data.get("images", []) if i["id"] == image_id), None)
    return {
        "image": image,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }


@router.post("/save")
async def save_annotations(request: SaveAnnotationsRequest):
    """Save complete COCO JSON data."""
    async with _get_coco_lock(request.coco_json_path):
        await _write_coco(request.coco_json_path, request.data)
    return {"status": "saved", "path": request.coco_json_path}


@router.post("/create")
async def create_annotation(request: CreateAnnotationRequest):
    """Create a new annotation."""
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        annotations = data.get("annotations", [])

        max_id = max((a["id"] for a in annotations), default=0)
        new_ann: Dict[str, Any] = {
            "id": max_id + 1,
            "image_id": request.image_id,
            "category_id": request.category_id,
            "bbox": request.bbox,
            "area": request.area or (request.bbox[2] * request.bbox[3]),
            "iscrowd": 0,
        }
        if request.segmentation:
            new_ann["segmentation"] = request.segmentation

        data["annotations"].append(new_ann)
        await _write_coco(request.coco_json_path, data)
    return new_ann


@router.put("/update")
async def update_annotation(request: UpdateAnnotationRequest):
    """Update an existing annotation."""
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        ann = next(
            (a for a in data.get("annotations", []) if a["id"] == request.annotation_id),
            None,
        )
        if not ann:
            raise HTTPException(404, f"Annotation {request.annotation_id} not found")

        if request.bbox is not None:
            ann["bbox"] = request.bbox
            ann["area"] = request.bbox[2] * request.bbox[3]
        if request.category_id is not None:
            ann["category_id"] = request.category_id
        if request.segmentation is not None:
            ann["segmentation"] = request.segmentation

        await _write_coco(request.coco_json_path, data)
    return ann


@router.post("/delete")
async def delete_annotation(request: DeleteAnnotationRequest):
    """Delete an annotation."""
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        data["annotations"] = [
            a for a in data.get("annotations", []) if a["id"] != request.annotation_id
        ]
        await _write_coco(request.coco_json_path, data)
    return {"status": "deleted", "annotation_id": request.annotation_id}


@router.post("/reannotate")
async def reannotate_with_sam3(request: ReannotateRequest):
    """Re-annotate an image region using SAM3."""
    data = await _read_coco(request.coco_json_path)
    ann = next(
        (a for a in data.get("annotations", []) if a["id"] == request.annotation_id),
        None,
    )
    if not ann:
        raise HTTPException(404, f"Annotation {request.annotation_id} not found")

    image = next(
        (i for i in data.get("images", []) if i["id"] == request.image_id), None
    )
    if not image:
        raise HTTPException(404, f"Image {request.image_id} not found")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{SEGMENTATION_SERVICE_URL}/sam3/segment-image",
                json={
                    "image_path": image.get("file_name", ""),
                    "bbox": ann.get("bbox"),
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                if result.get("segmentation"):
                    async with _get_coco_lock(request.coco_json_path):
                        # Re-read to avoid overwriting concurrent edits
                        data = await _read_coco(request.coco_json_path)
                        target = next(
                            (a for a in data.get("annotations", []) if a["id"] == request.annotation_id),
                            None,
                        )
                        if target:
                            target["segmentation"] = result["segmentation"]
                            await _write_coco(request.coco_json_path, data)
                            ann = target
                return {"status": "reannotated", "annotation": ann}
            raise HTTPException(resp.status_code, "SAM3 reannotation failed")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")


@router.post("/segment-bbox")
async def segment_bbox(request: SegmentBboxRequest):
    """Get SAM3 segmentation mask for a user-drawn bounding box.

    Returns the COCO polygon without touching any file — the caller decides
    whether to attach the segmentation to an annotation.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SEGMENTATION_SERVICE_URL}/sam3/segment-image",
                json={
                    "image_path": request.image_path,
                    "bbox": request.bbox,
                    "text_prompt": request.text_hint,
                    "return_polygon": True,
                    "return_mask": False,
                    "simplify_polygon": True,
                    "simplify_tolerance": 2.0,
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": result.get("success", False),
                    "segmentation_coco": result.get("segmentation_coco") if result.get("success") else None,
                }
            return {"success": False, "segmentation_coco": None}
    except httpx.RequestError as e:
        logger.warning(f"SAM3 segment-bbox unavailable: {e}")
        return {"success": False, "segmentation_coco": None}


@router.post("/segment-point")
async def segment_point(request: SegmentPointRequest):
    """Get SAM3 segmentation mask for user-clicked points (Sam3TrackerModel).

    Returns the COCO polygon without touching any file — the caller decides
    whether to create an annotation from the result.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SEGMENTATION_SERVICE_URL}/sam3/segment-point",
                json={
                    "image_path": request.image_path,
                    "points": request.points,
                    "labels": request.labels,
                    "text_hint": request.text_hint,
                    "return_polygon": request.return_polygon,
                    "simplify_polygon": request.simplify_polygon,
                    "simplify_tolerance": request.simplify_tolerance,
                },
            )
            if resp.status_code == 200:
                return resp.json()
            return {"success": False, "segmentation_coco": None, "bbox": None, "confidence": 0.0}
    except httpx.RequestError as e:
        logger.warning(f"SAM3 segment-point unavailable: {e}")
        return {"success": False, "segmentation_coco": None, "bbox": None, "confidence": 0.0}


class SegmentTextRequest(BaseModel):
    image_path: str
    text_prompt: str


@router.post("/segment-text")
async def segment_text(request: SegmentTextRequest):
    """Get SAM3 segmentation mask for a free-text prompt (PCS mode).

    Returns the best-matching COCO polygon for the given text in the image.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{SEGMENTATION_SERVICE_URL}/sam3/segment-image",
                json={
                    "image_path": request.image_path,
                    "text_prompt": request.text_prompt,
                    "return_polygon": True,
                    "return_mask": False,
                    "simplify_polygon": True,
                    "simplify_tolerance": 2.0,
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": result.get("success", False),
                    "segmentation_coco": result.get("segmentation_coco") if result.get("success") else None,
                    "bbox": result.get("bbox"),
                    "confidence": result.get("confidence", 0.0),
                    "error": result.get("error"),
                }
            return {"success": False, "segmentation_coco": None, "bbox": None, "confidence": 0.0}
    except httpx.RequestError as e:
        logger.warning(f"SAM3 segment-text unavailable: {e}")
        return {"success": False, "segmentation_coco": None, "bbox": None, "confidence": 0.0}


# NOTE: POST /annotations/export was replaced by the multi-format engine at
# POST /export/start (see app/routers/export.py).
