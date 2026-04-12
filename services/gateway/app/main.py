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

from app.routers import annotations, comparison, labeling, datasets, filesystem, sam3

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
    os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://frontend:80").split(",")
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


@app.get("/jobs", tags=["Jobs"])
async def get_all_jobs():
    """Unified endpoint that aggregates labeling and SAM3 jobs."""
    import httpx as _httpx

    all_jobs = []

    async with _httpx.AsyncClient(timeout=10.0) as client:
        # Fetch labeling jobs
        try:
            resp = await client.get(f"{SEGMENTATION_SERVICE_URL}/labeling/jobs")
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", []) if isinstance(data, dict) else []
                for j in jobs:
                    j["source"] = "labeling"
                all_jobs.extend(jobs)
        except Exception:
            pass

        # Fetch SAM3 jobs
        try:
            resp = await client.get(f"{SEGMENTATION_SERVICE_URL}/sam3/jobs")
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", []) if isinstance(data, dict) else []
                for j in jobs:
                    j["source"] = "sam3"
                all_jobs.extend(jobs)
        except Exception:
            pass

    all_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"jobs": all_jobs, "total": len(all_jobs)}


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
