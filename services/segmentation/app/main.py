"""
Segmentation Service
====================
FastAPI microservice for semantic scene analysis and SAM3 segmentation.

Uses SAM3 (Segment Anything Model 3) for Promptable Concept Segmentation (PCS).
SAM3 released 2025-11-19 - segments all instances matching a text concept.

Model: facebook/sam3 (848M parameters)
"""

import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import HealthResponse
from app.service_state import (
    state, init_gpu, init_sam3_background, _load_sam3_sync,
    JOB_DATABASE_AVAILABLE, get_job_db,
)
from app.routers import analysis, extraction, sam3_tool, labeling, video, propagate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic with fast non-blocking startup."""
    logger.info("Starting Segmentation Service...")
    startup_start = time.time()

    # Initialize database for job persistence
    if JOB_DATABASE_AVAILABLE and get_job_db is not None:
        try:
            state.db = get_job_db()
            logger.info("Job database initialized for persistence")
            orphaned = state.db.mark_orphaned_jobs("segmentation")
            if orphaned:
                logger.warning(f"Marked {orphaned} orphaned segmentation jobs as interrupted")
        except Exception as e:
            logger.warning(f"Failed to initialize job database: {e}")
            state.db = None
    else:
        logger.warning("Job database not available - jobs will not persist across restarts")

    # Fast startup: Initialize GPU detection only (non-blocking)
    init_gpu()

    use_lazy_load = os.environ.get("SAM3_LAZY_LOAD", "true").lower() != "false"

    if use_lazy_load:
        logger.info("Service starting with lazy SAM3 loading (non-blocking)")
        state._loading_task = asyncio.create_task(init_sam3_background())
    else:
        logger.info("Service starting with blocking SAM3 loading")
        _load_sam3_sync()

    startup_time = time.time() - startup_start
    logger.info(f"Segmentation Service ready in {startup_time:.1f}s (SAM3 loading in background: {use_lazy_load})")
    yield

    # Cleanup
    logger.info("Shutting down Segmentation Service...")
    if state._loading_task and not state._loading_task.done():
        state._loading_task.cancel()
        try:
            await state._loading_task
        except asyncio.CancelledError:
            pass

    state.sam3_model = None
    state.scene_analyzer = None


# Create FastAPI app
app = FastAPI(
    title="Segmentation Service",
    description="Semantic scene analysis and SAM3 segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
ALLOWED_ORIGINS = [
    o.strip() for o in
    os.environ.get("CORS_ORIGINS", "http://localhost:3001,http://localhost:5173,http://gateway:8000").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(extraction.router, tags=["Object Extraction"])
app.include_router(sam3_tool.router, tags=["SAM3 Tool"])
app.include_router(labeling.router, tags=["Labeling Tool"])
app.include_router(video.router, tags=["Video"])
app.include_router(propagate.router, tags=["Propagation"])


# ---------------------------------------------------------------------------
# Health endpoints (kept in main for simplicity)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy",
        sam3_available=state.sam3_available,
        sam3_loading=state.sam3_loading,
        sam3_load_progress=state.sam3_load_progress,
        sam3_load_error=state.sam3_load_error,
        sam3_model_id=state.sam3_model_id,
        gpu_available=state.gpu_available,
        model_loaded=state.scene_analyzer is not None,
        version="1.0.0",
    )


@app.get("/model-status", tags=["Health"])
async def model_status():
    """Get detailed model loading status."""
    return {
        "sam3": {
            "available": state.sam3_available,
            "loading": state.sam3_loading,
            "progress": state.sam3_load_progress,
            "error": state.sam3_load_error,
            "model_id": state.sam3_model_id,
        },
        "gpu": {
            "available": state.gpu_available,
            "device": state.device,
        },
        "scene_analyzer": {"initialized": state.scene_analyzer is not None},
        "object_extractor": {"initialized": state.object_extractor is not None},
    }


@app.get("/ping")
async def ping():
    return {"status": "ok"}
