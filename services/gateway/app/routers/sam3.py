"""
SAM3 Tool Router - Proxy to segmentation service for SAM3 operations.
"""

import os
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SAM3"], prefix="/sam3")

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


class ConvertDatasetRequest(BaseModel):
    coco_json_path: str
    images_dir: str
    output_dir: str
    min_area: int = Field(default=100, ge=10)
    confidence_threshold: float = Field(default=0.8, ge=0.1, le=1.0)


async def _proxy(method: str, path: str, data=None, params=None):
    url = f"{SEGMENTATION_SERVICE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=data)
            else:
                raise HTTPException(405, "Method not allowed")

            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, resp.text)
            return resp.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")


@router.post("/convert-dataset")
async def convert_dataset(request: ConvertDatasetRequest):
    return await _proxy("POST", "/sam3/convert-dataset", request.model_dump())


@router.get("/jobs")
async def list_sam3_jobs():
    return await _proxy("GET", "/sam3/jobs")


@router.get("/jobs/{job_id}")
async def get_sam3_job_status(job_id: str):
    return await _proxy("GET", f"/sam3/jobs/{job_id}")
