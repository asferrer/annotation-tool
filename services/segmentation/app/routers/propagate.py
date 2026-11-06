"""
Propagation endpoints: INSID3 one-shot mask propagation over a dataset.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.insid3.engine import cancel_flags, run_propagation_job
from app.insid3.loader import weights_status
from app.insid3.schemas import PropagateJobResponse, PropagateRequest
from app.model_manager import ModelBusyError, manager
from app.service_state import state, thread_pool

try:
    from shared.job_lifecycle import JobLifecycle
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, "/app")
    from shared.job_lifecycle import JobLifecycle

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/propagate/status", tags=["Propagation"])
async def propagate_status():
    """INSID3 readiness: DINOv3 weights presence and manager state."""
    return {"insid3": weights_status(), "manager": manager.status()}


@router.post("/propagate/start", response_model=PropagateJobResponse, tags=["Propagation"])
async def start_propagation(request: PropagateRequest):
    if not state.gpu_available:
        raise HTTPException(503, "GPU not available for INSID3 propagation")

    insid3 = weights_status()
    if not insid3["weights_present"]:
        raise HTTPException(424, insid3["instructions"])

    if not Path(request.reference_image_path).exists():
        raise HTTPException(404, f"Reference image not found: {request.reference_image_path}")
    if not Path(request.target_directory).is_dir():
        raise HTTPException(404, f"Target directory not found: {request.target_directory}")

    status = manager.status()
    active = status["active_model"]
    if active and active != "insid3" and status["refcounts"].get(active, 0) > 0:
        raise HTTPException(409, f"Model '{active}' has active jobs; retry when they finish")

    job_id = f"propagate_{uuid.uuid4().hex[:12]}"
    lifecycle = JobLifecycle.create(
        job_id, job_type="propagate", service="segmentation",
        request_params=request.model_dump(), output_path=request.output_json,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(thread_pool, _run_job_safe, job_id, request, lifecycle)
    return PropagateJobResponse(
        success=True, job_id=job_id, status="queued",
        message=f"Propagating '{request.label}' over {request.target_directory}",
    )


def _run_job_safe(job_id: str, request: PropagateRequest, lifecycle) -> None:
    try:
        run_propagation_job(job_id, request, lifecycle)
    except ModelBusyError as e:
        lifecycle.fail(str(e))
    except Exception as e:
        logger.exception(f"Unexpected propagation failure {job_id}")
        lifecycle.fail(str(e))


@router.get("/propagate/jobs/{job_id}", tags=["Propagation"])
async def get_propagation_job(job_id: str):
    if not state.db:
        raise HTTPException(503, "Job database unavailable")
    job = await asyncio.to_thread(state.db.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.delete("/propagate/jobs/{job_id}", tags=["Propagation"])
async def cancel_propagation_job(job_id: str):
    cancel_flags[job_id] = True
    return {"status": "cancel_requested", "job_id": job_id}
