"""
Labeling tool endpoints: auto-labeling, relabeling and job management.

Each sub-module owns one concern and exposes its own ``APIRouter``; this package
combines them into a single ``router`` so the public API stays unchanged.
"""

from fastapi import APIRouter

from app.routers.labeling import jobs, relabel, resume, start

router = APIRouter()
router.include_router(start.router)
router.include_router(relabel.router)
router.include_router(jobs.router)
router.include_router(resume.router)
