"""
SAM3 Router - Proxy to segmentation service SAM3 tool endpoints.

Passthrough proxies: request validation lives in the segmentation service
schemas (single source of truth), so bodies are forwarded as-is.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body

from app.proxy import proxy_segmentation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SAM3"], prefix="/sam3")


@router.post("/segment-image")
async def segment_image(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/sam3/segment-image", body)


@router.post("/segment-point")
async def segment_point(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/sam3/segment-point", body)


@router.post("/convert-dataset")
async def convert_dataset(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/sam3/convert-dataset", body)


@router.get("/jobs")
async def list_jobs() -> Any:
    return await proxy_segmentation("GET", "/sam3/jobs")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> Any:
    return await proxy_segmentation("GET", f"/sam3/jobs/{job_id}")
