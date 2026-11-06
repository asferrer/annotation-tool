"""Pydantic schemas for INSID3 propagation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PropagateRequest(BaseModel):
    reference_image_path: str
    reference_polygon: List[List[float]] = Field(..., min_length=1)
    target_directory: str
    label: str = Field(..., min_length=1)
    merge_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    min_area: int = Field(default=100, ge=1)
    output_json: str
    max_images: Optional[int] = Field(default=None, ge=1)
    include_reference: bool = False


class PropagateJobResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str = ""
    total_images: int = 0
    error: Optional[str] = None
