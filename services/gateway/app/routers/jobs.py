"""
Jobs Router - Aggregated job listing across services.

Merges jobs persisted in the shared SQLite database with the in-memory
job lists exposed by the segmentation service, deduplicating by job id.
"""

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter

from app.proxy import proxy_segmentation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Jobs"], prefix="/jobs")

try:
    from shared.job_database import get_job_db
    JOB_DB_AVAILABLE = True
except ImportError:  # pragma: no cover - shared module always bundled in image
    JOB_DB_AVAILABLE = False


def _db_jobs_sync() -> List[Dict[str, Any]]:
    """Read persisted jobs from the shared database, normalized to job_id."""
    if not JOB_DB_AVAILABLE:
        return []
    try:
        rows = get_job_db().list_jobs(limit=100)
    except Exception as e:
        logger.warning(f"Job database unavailable: {e}")
        return []

    jobs = []
    for row in rows:
        total = row.get("total_items") or 0
        processed = row.get("processed_items") or 0
        progress = round(processed / total * 100, 1) if total > 0 else 0.0
        jobs.append({
            "job_id": row.get("id"),
            "type": row.get("job_type", "unknown"),
            "job_type": row.get("job_type", "unknown"),
            "service": row.get("service", "unknown"),
            "status": row.get("status", "unknown"),
            "progress": progress,
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "total_images": total,
            "processed_images": processed,
            "output_dir": row.get("output_path") or "",
            "error_message": row.get("error_message"),
        })
    return jobs


async def _segmentation_jobs(path: str, job_type: str) -> List[Dict[str, Any]]:
    """Fetch a job list from the segmentation service, tolerating downtime."""
    try:
        data = await proxy_segmentation("GET", path, timeout=10.0)
    except Exception as e:
        logger.warning(f"Could not fetch {path}: {e}")
        return []

    raw = data.get("jobs", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    return [{**job, "type": job.get("type", job_type)} for job in raw if isinstance(job, dict)]


@router.get("")
async def list_all_jobs() -> Dict[str, Any]:
    """Aggregate jobs from the shared DB and the segmentation service."""
    db_jobs, labeling, extraction, sam3 = await asyncio.gather(
        asyncio.to_thread(_db_jobs_sync),
        _segmentation_jobs("/labeling/jobs", "labeling"),
        _segmentation_jobs("/extract/jobs", "extraction"),
        _segmentation_jobs("/sam3/jobs", "sam3_conversion"),
    )

    merged: Dict[str, Dict[str, Any]] = {}
    # In-memory jobs from segmentation are the freshest: they win over DB rows.
    for job in db_jobs + labeling + extraction + sam3:
        job_id = job.get("job_id") or job.get("id")
        if not job_id:
            continue
        merged[job_id] = {**merged.get(job_id, {}), **job, "job_id": job_id}

    jobs = sorted(
        merged.values(),
        key=lambda j: j.get("created_at") or "",
        reverse=True,
    )
    return {"jobs": jobs, "total": len(jobs)}
