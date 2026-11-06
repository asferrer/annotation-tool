"""
Annotation Tool Gateway
=======================
Lightweight API gateway for the annotation tool.
Proxies requests to the SAM3 segmentation service and handles
COCO annotation CRUD operations.
"""

import os
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import annotations, comparison, labeling, datasets, filesystem, sam3, jobs, export, videos, tracks, propagate

logger = logging.getLogger(__name__)

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Annotation Tool Gateway starting...")
    yield
    logger.info("Annotation Tool Gateway shutting down...")


app = FastAPI(
    title="Annotation Tool Gateway",
    description="AI-powered annotation and labeling tool with SAM3",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = [
    o.strip() for o in
    os.environ.get("CORS_ORIGINS", "http://localhost:3001,http://localhost:5173,http://frontend:80").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(annotations.router)
app.include_router(comparison.router)
app.include_router(labeling.router)
app.include_router(datasets.router)
app.include_router(filesystem.router)
app.include_router(sam3.router)
app.include_router(jobs.router)
app.include_router(export.router)
app.include_router(videos.router)
app.include_router(tracks.router)
app.include_router(propagate.router)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/health")
async def health():
    services = {}
    # Check segmentation service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SEGMENTATION_SERVICE_URL}/health")
            services["segmentation"] = resp.json() if resp.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        services["segmentation"] = {"status": "unreachable"}

    all_healthy = all(
        s.get("status") == "healthy" for s in services.values()
    )
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services,
    }
