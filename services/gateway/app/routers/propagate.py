"""
Propagate Router - proxy to segmentation INSID3 propagation endpoints.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body

from app.proxy import proxy_segmentation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Propagation"], prefix="/propagate")


@router.get("/status")
async def status() -> Any:
    return await proxy_segmentation("GET", "/propagate/status")


@router.post("/start")
async def start(body: Dict[str, Any] = Body(...)) -> Any:
    return await proxy_segmentation("POST", "/propagate/start", body)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Any:
    return await proxy_segmentation("GET", f"/propagate/jobs/{job_id}")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> Any:
    return await proxy_segmentation("DELETE", f"/propagate/jobs/{job_id}")
