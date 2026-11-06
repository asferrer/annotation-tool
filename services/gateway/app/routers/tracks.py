"""
Tracks Router - edit video tracks stored in COCO-extended JSON.

Pure JSON operations (no GPU): list per-track summaries, rename a track's
category, delete a track, merge two tracks.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.routers.annotations import _get_coco_lock, _read_coco, _write_coco

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Tracks"], prefix="/tracks")


class TrackActionRequest(BaseModel):
    coco_json_path: str
    track_id: int


class RenameTrackRequest(TrackActionRequest):
    new_label: str


class MergeTracksRequest(BaseModel):
    coco_json_path: str
    source_track_id: int
    target_track_id: int


def _track_annotations(data: Dict[str, Any], track_id: int) -> List[Dict[str, Any]]:
    return [a for a in data.get("annotations", []) if a.get("track_id") == track_id]


@router.get("")
async def list_tracks(coco_json_path: str = Query(...)) -> Dict[str, Any]:
    """Summarize tracks: frames covered, category, score stats."""
    data = await _read_coco(coco_json_path)
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    image_frames = {img["id"]: img.get("frame_index", img["id"] - 1) for img in data.get("images", [])}

    tracks: Dict[int, Dict[str, Any]] = {}
    for ann in data.get("annotations", []):
        track_id = ann.get("track_id")
        if track_id is None:
            continue
        frame = image_frames.get(ann.get("image_id"), 0)
        entry = tracks.setdefault(track_id, {
            "track_id": track_id,
            "category_id": ann.get("category_id"),
            "label": categories.get(ann.get("category_id"), "unknown"),
            "frame_count": 0,
            "first_frame": frame,
            "last_frame": frame,
            "avg_score": 0.0,
            "_score_sum": 0.0,
        })
        entry["frame_count"] += 1
        entry["first_frame"] = min(entry["first_frame"], frame)
        entry["last_frame"] = max(entry["last_frame"], frame)
        entry["_score_sum"] += float(ann.get("score", 0.0))

    result = []
    for entry in tracks.values():
        entry["avg_score"] = round(entry.pop("_score_sum") / max(entry["frame_count"], 1), 4)
        result.append(entry)
    result.sort(key=lambda t: t["track_id"])
    return {"tracks": result, "total": len(result), "videos": data.get("videos", [])}


@router.post("/rename")
async def rename_track(request: RenameTrackRequest) -> Dict[str, Any]:
    """Reassign a track to a category by name (created if missing)."""
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        anns = _track_annotations(data, request.track_id)
        if not anns:
            raise HTTPException(404, f"Track {request.track_id} not found")

        label = request.new_label.strip()
        category = next((c for c in data["categories"] if c["name"] == label), None)
        if category is None:
            new_id = max((c["id"] for c in data["categories"]), default=0) + 1
            category = {"id": new_id, "name": label}
            data["categories"].append(category)

        for ann in anns:
            ann["category_id"] = category["id"]
        await _write_coco(request.coco_json_path, data)
    return {"status": "renamed", "track_id": request.track_id, "category": category}


@router.post("/delete")
async def delete_track(request: TrackActionRequest) -> Dict[str, Any]:
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        before = len(data.get("annotations", []))
        data["annotations"] = [
            a for a in data.get("annotations", []) if a.get("track_id") != request.track_id
        ]
        removed = before - len(data["annotations"])
        if removed == 0:
            raise HTTPException(404, f"Track {request.track_id} not found")
        await _write_coco(request.coco_json_path, data)
    return {"status": "deleted", "track_id": request.track_id, "removed_annotations": removed}


@router.post("/merge")
async def merge_tracks(request: MergeTracksRequest) -> Dict[str, Any]:
    """Reassign all annotations of source track to the target track id."""
    if request.source_track_id == request.target_track_id:
        raise HTTPException(422, "source and target tracks must differ")
    async with _get_coco_lock(request.coco_json_path):
        data = await _read_coco(request.coco_json_path)
        source = _track_annotations(data, request.source_track_id)
        target = _track_annotations(data, request.target_track_id)
        if not source or not target:
            raise HTTPException(404, "Source or target track not found")

        target_category: Optional[int] = target[0].get("category_id")
        for ann in source:
            ann["track_id"] = request.target_track_id
            if target_category is not None:
                ann["category_id"] = target_category
        await _write_coco(request.coco_json_path, data)
    return {
        "status": "merged",
        "target_track_id": request.target_track_id,
        "moved_annotations": len(source),
    }
