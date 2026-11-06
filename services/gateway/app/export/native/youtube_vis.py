"""
Native YouTube-VIS exporter (video instance segmentation).

One annotation per track with per-frame lists (null where absent):
{videos, annotations: [{id, video_id, category_id, segmentations, bboxes, areas, ...}], categories}
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def _frame_index_map(coco: Dict[str, Any]) -> Dict[int, int]:
    """image_id -> frame position (0-based) using frame_index when present."""
    images = sorted(
        coco.get("images", []),
        key=lambda img: img.get("frame_index", img["id"] - 1),
    )
    return {img["id"]: pos for pos, img in enumerate(images)}


def export_youtube_vis(coco: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_of = _frame_index_map(coco)
    num_frames = len(frame_of)

    videos = coco.get("videos") or [{
        "id": 1, "name": "video", "width": 0, "height": 0, "fps": 0,
    }]
    vis_videos = [{
        "id": v.get("id", 1),
        "width": v.get("width", 0),
        "height": v.get("height", 0),
        "length": num_frames,
        "file_names": [
            Path(img["file_name"]).name
            for img in sorted(coco.get("images", []), key=lambda i: frame_of[i["id"]])
        ],
    } for v in videos]

    tracks: Dict[int, Dict[str, Any]] = {}
    for ann in coco.get("annotations", []):
        track_id = ann.get("track_id")
        if track_id is None or ann.get("image_id") not in frame_of:
            continue
        frame = frame_of[ann["image_id"]]
        entry = tracks.setdefault(track_id, {
            "id": track_id,
            "video_id": ann.get("video_id", 1),
            "category_id": ann["category_id"],
            "iscrowd": 0,
            "segmentations": [None] * num_frames,
            "bboxes": [None] * num_frames,
            "areas": [None] * num_frames,
            "_scores": [],
        })
        entry["segmentations"][frame] = ann.get("segmentation") or None
        entry["bboxes"][frame] = ann.get("bbox")
        entry["areas"][frame] = ann.get("area")
        if "score" in ann:
            entry["_scores"].append(float(ann["score"]))

    annotations: List[Dict[str, Any]] = []
    for entry in tracks.values():
        scores = entry.pop("_scores")
        if scores:
            entry["score"] = round(sum(scores) / len(scores), 4)
        annotations.append(entry)

    result = {
        "info": {"description": "YouTube-VIS export", "source": "annotation-tool"},
        "videos": vis_videos,
        "annotations": sorted(annotations, key=lambda a: a["id"]),
        "categories": coco.get("categories", []),
    }
    (out_dir / "youtube_vis.json").write_text(json.dumps(result, indent=2))
    return out_dir
