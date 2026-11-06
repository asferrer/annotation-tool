"""Pydantic schemas for the video pipeline."""

from typing import List, Optional

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


class ProbeRequest(BaseModel):
    video_path: str


class ExtractFramesRequest(BaseModel):
    video_path: str
    target_fps: Optional[float] = Field(default=None, gt=0)
    frame_stride: Optional[int] = Field(default=None, ge=1)
    max_frames: Optional[int] = Field(default=None, ge=1)
    output_dir: Optional[str] = None


class ExtractFramesResponse(BaseModel):
    frames_dir: str
    extracted_frames: int
    frame_stride: int
    effective_fps: float
    video: VideoInfo


class TrackRequest(BaseModel):
    frames_dir: str
    prompts: List[str] = Field(..., min_length=1)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    output_json: str
    video_name: Optional[str] = None
    source_video_path: Optional[str] = None
    fps: Optional[float] = None
    frame_stride: int = 1
    max_frames: Optional[int] = Field(default=None, ge=1)


class TrackJobResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str = ""
    total_frames: int = 0
    error: Optional[str] = None
