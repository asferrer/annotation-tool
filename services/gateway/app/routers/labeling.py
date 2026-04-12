"""
Labeling Router - Proxy to segmentation service for auto-labeling.
"""

import os
import logging
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Labeling"], prefix="/labeling")

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


# ---------- Schemas ----------

class LabelingStartRequest(BaseModel):
    image_directories: List[str] = Field(..., min_length=1)
    classes: List[str] = Field(..., min_length=1)
    class_mapping: Optional[Dict[str, str]] = None
    output_dir: str
    output_formats: List[str] = Field(default=["coco"])
    task_type: str = Field(default="segmentation")
    min_confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    min_area: int = Field(default=100, ge=10)
    max_instances_per_image: int = Field(default=100, ge=1, le=1000)
    simplify_polygons: bool = True
    save_visualizations: bool = True
    padding: int = Field(default=0, ge=0, le=50)
    preview_mode: bool = False
    preview_count: int = Field(default=20, ge=5, le=100)
    deduplication_strategy: str = Field(default="confidence")


class RelabelingRequest(BaseModel):
    image_directories: List[str] = Field(..., min_length=1)
    output_dir: str
    relabel_mode: str = Field(default="add")
    new_classes: Optional[List[str]] = None
    min_confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    coco_json_path: Optional[str] = None
    output_formats: List[str] = Field(default=["coco"])
    task_type: str = Field(default="segmentation")
    simplify_polygons: bool = True
    preview_mode: bool = False
    preview_count: int = Field(default=20, ge=5, le=100)
    deduplication_strategy: str = Field(default="confidence")


# ---------- Proxy helper ----------

async def _proxy(method: str, path: str, data=None, params=None):
    url = f"{SEGMENTATION_SERVICE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=data)
            elif method == "DELETE":
                resp = await client.delete(url, params=params)
            else:
                raise HTTPException(405, "Method not allowed")

            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, resp.text)
            return resp.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")


# ---------- Endpoints ----------

@router.post("/start")
async def start_labeling(request: LabelingStartRequest):
    return await _proxy("POST", "/labeling/start", request.model_dump())


@router.post("/relabel")
async def start_relabeling(request: RelabelingRequest):
    return await _proxy("POST", "/labeling/relabel", request.model_dump())


@router.get("/jobs")
async def list_jobs():
    return await _proxy("GET", "/labeling/jobs")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    return await _proxy("GET", f"/labeling/jobs/{job_id}")


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    return await _proxy("GET", f"/labeling/jobs/{job_id}/result")


@router.get("/jobs/{job_id}/previews")
async def get_job_previews(job_id: str):
    return await _proxy("GET", f"/labeling/jobs/{job_id}/previews")


@router.get("/jobs/{job_id}/partial-annotations")
async def get_partial_annotations(job_id: str):
    return await _proxy("GET", f"/labeling/jobs/{job_id}/partial-annotations")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    return await _proxy("DELETE", f"/labeling/jobs/{job_id}")
