"""
Export Router - multi-format dataset export/import jobs (Datumaro engine).
"""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.export.engine import run_export
from app.export.formats import list_formats
from app.export.importer import run_import
from app.routers.filesystem import _validate_path
from shared.job_database import get_job_db
from shared.job_lifecycle import JobLifecycle

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Export"])

EXPORTS_BASE = Path(os.environ.get("EXPORTS_BASE", "/app/datasets/exports"))

# Datumaro conversions are CPU-bound; bound their concurrency.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="export")


# ---------- Schemas ----------

class ExportStartRequest(BaseModel):
    coco_json_path: str
    format: str
    include_images: bool = False
    images_dir: Optional[str] = None


class ImportStartRequest(BaseModel):
    input_path: str
    format: str
    task_name: str = Field(..., min_length=1, max_length=100)


# ---------- Job runners (worker threads) ----------

def _run_export_job(job_id: str, request: ExportStartRequest) -> None:
    lifecycle = JobLifecycle(job_id)
    lifecycle.start(f"Exporting {request.coco_json_path} to {request.format}")
    try:
        out_dir = EXPORTS_BASE / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = run_export(
            request.coco_json_path,
            request.format,
            out_dir,
            include_images=request.include_images,
            images_dir=request.images_dir,
        )
        lifecycle.db.update_job_output(job_id, str(zip_path))
        lifecycle.complete({"zip_path": str(zip_path), "format": request.format})
    except Exception as e:
        logger.exception(f"Export job {job_id} failed")
        lifecycle.fail(str(e))


def _run_import_job(job_id: str, request: ImportStartRequest) -> None:
    lifecycle = JobLifecycle(job_id)
    lifecycle.start(f"Importing {request.input_path} as {request.format}")
    try:
        summary = run_import(request.input_path, request.format, request.task_name)
        lifecycle.db.update_job_output(job_id, summary["coco_json_path"])
        lifecycle.complete(summary)
    except Exception as e:
        logger.exception(f"Import job {job_id} failed")
        lifecycle.fail(str(e))


# ---------- Endpoints ----------

@router.get("/export/formats")
async def get_formats():
    return [spec.model_dump() for spec in list_formats()]


@router.post("/export/start")
async def start_export(request: ExportStartRequest):
    _validate_path(request.coco_json_path)
    if request.images_dir:
        _validate_path(request.images_dir)

    job_id = f"export_{uuid.uuid4().hex[:12]}"
    JobLifecycle.create(job_id, job_type="export", service="gateway",
                        request_params=request.model_dump(), total_items=1)
    asyncio.get_running_loop().run_in_executor(_executor, _run_export_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@router.post("/import/start")
async def start_import(request: ImportStartRequest):
    _validate_path(request.input_path)

    job_id = f"import_{uuid.uuid4().hex[:12]}"
    JobLifecycle.create(job_id, job_type="import", service="gateway",
                        request_params=request.model_dump(), total_items=1)
    asyncio.get_running_loop().run_in_executor(_executor, _run_import_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@router.get("/export/jobs/{job_id}")
async def get_export_job(job_id: str):
    job = await asyncio.to_thread(lambda: get_job_db().get_job(job_id))
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.get("/export/jobs/{job_id}/download")
async def download_export(job_id: str):
    job = await asyncio.to_thread(lambda: get_job_db().get_job(job_id))
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.get("status") != "completed":
        raise HTTPException(409, f"Job not completed (status: {job.get('status')})")

    zip_path = Path(job.get("output_path") or "")
    if not zip_path.exists():
        raise HTTPException(404, "Export artifact no longer exists")
    request_params = job.get("request_params") or {}
    fmt = request_params.get("format", "export") if isinstance(request_params, dict) else "export"
    return FileResponse(zip_path, filename=f"{fmt}_{job_id}.zip", media_type="application/zip")
