"""
SAM3 Video tracking job: frames dir + text prompts -> COCO-extended JSON.

Output is standard COCO where each frame is an image (extra fields video_id,
frame_index), plus a top-level `videos` block, and each annotation carries a
persistent `track_id` (the SAM3 Video object id).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .schemas import TrackRequest

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 25
POLYGON_EPSILON = 2.0

# job_id -> True when a cancel was requested
cancel_flags: Dict[str, bool] = {}


def mask_to_polygons(mask: np.ndarray, epsilon: float = POLYGON_EPSILON) -> List[List[float]]:
    """Binary mask -> COCO polygons via contour extraction."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        polygons.append([float(v) for point in approx.reshape(-1, 2) for v in point])
    return polygons


def _list_frames(frames_dir: str) -> List[Path]:
    frames = sorted(Path(frames_dir).glob("*.jpg")) or sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    return frames


def _base_coco(request: TrackRequest, frames: List[Path], width: int, height: int) -> Dict[str, Any]:
    return {
        "info": {"description": "SAM3 Video tracking", "source": "annotation-tool"},
        "videos": [{
            "id": 1,
            "name": request.video_name or Path(request.frames_dir).parent.name,
            "source_path": request.source_video_path or "",
            "fps": request.fps or 0.0,
            "frame_stride": request.frame_stride,
            "frame_count": len(frames),
            "width": width,
            "height": height,
        }],
        "images": [
            {
                "id": idx + 1,
                "file_name": str(frame),
                "width": width,
                "height": height,
                "video_id": 1,
                "frame_index": idx,
            }
            for idx, frame in enumerate(frames)
        ],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": prompt} for i, prompt in enumerate(request.prompts)
        ],
    }


def _annotations_for_frame(
    processed: Dict[str, Any],
    frame_index: int,
    prompt_category: Dict[str, int],
    obj_to_category: Dict[int, int],
    min_confidence: float,
    next_ann_id: int,
) -> List[Dict[str, Any]]:
    """Convert one frame's postprocessed outputs into COCO annotations."""
    for prompt, obj_ids in (processed.get("prompt_to_obj_ids") or {}).items():
        for obj_id in obj_ids:
            obj_to_category.setdefault(int(obj_id), prompt_category.get(prompt, 1))

    annotations = []
    object_ids = processed["object_ids"].tolist()
    scores = processed["scores"].tolist()
    boxes = processed["boxes"].tolist()
    masks = processed["masks"]

    for i, obj_id in enumerate(object_ids):
        score = float(scores[i])
        if score < min_confidence:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        mask_np = masks[i].cpu().numpy() if hasattr(masks[i], "cpu") else np.asarray(masks[i])
        polygons = mask_to_polygons(mask_np > 0.5)
        annotations.append({
            "id": next_ann_id + len(annotations),
            "image_id": frame_index + 1,
            "category_id": obj_to_category.get(int(obj_id), 1),
            "bbox": [round(v, 2) for v in bbox],
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
            "segmentation": polygons,
            "score": round(score, 4),
            "track_id": int(obj_id),
            "video_id": 1,
            "source": "sam3_video",
        })
    return annotations


def run_tracking_job(job_id: str, request: TrackRequest, lifecycle) -> None:
    """Worker-thread entrypoint: full tracking pipeline with checkpointing."""
    import torch
    from app.model_manager import manager
    from app.service_state import state

    cancel_flags.pop(job_id, None)
    frames = _list_frames(request.frames_dir)
    if request.max_frames:
        frames = frames[: request.max_frames]

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    coco = _base_coco(request, frames, width, height)
    prompt_category = {c["name"]: c["id"] for c in coco["categories"]}
    obj_to_category: Dict[int, int] = {}
    output_path = Path(request.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    handle = manager.acquire("sam3_video")
    try:
        model, processor = handle["model"], handle["processor"]
        lifecycle.start(f"Tracking {len(frames)} frames with prompts {request.prompts}")

        # A list of PIL images is read as ONE video's frames (a list of str paths
        # would be parsed as a batch of videos via torchvision.io.read_video).
        from PIL import Image

        pil_frames = [Image.open(str(f)).convert("RGB") for f in frames]
        session = processor.init_video_session(
            video=pil_frames,
            inference_device=state.device,
            video_storage_device="cpu",
            dtype=handle.get("dtype"),
        )
        del pil_frames
        processor.add_text_prompt(session, request.prompts)

        next_ann_id = 1
        with torch.inference_mode():
            iterator = model.propagate_in_video_iterator(
                session, start_frame_idx=0, max_frame_num_to_track=len(frames),
            )
            for frame_index, output in enumerate(iterator):
                if cancel_flags.get(job_id):
                    lifecycle.cancel({"processed_frames": frame_index})
                    return
                processed = processor.postprocess_outputs(session, output)
                anns = _annotations_for_frame(
                    processed, frame_index, prompt_category, obj_to_category,
                    request.min_confidence, next_ann_id,
                )
                coco["annotations"].extend(anns)
                next_ann_id += len(anns)

                if (frame_index + 1) % CHECKPOINT_EVERY == 0 or frame_index == len(frames) - 1:
                    output_path.write_text(json.dumps(coco))
                    lifecycle.update_progress(
                        frame_index + 1,
                        current_item=f"frame {frame_index + 1}/{len(frames)}",
                        details={"annotations": len(coco["annotations"])},
                    )

        output_path.write_text(json.dumps(coco, indent=2))
        track_ids = {a["track_id"] for a in coco["annotations"]}
        lifecycle.db.update_job_output(job_id, str(output_path))
        lifecycle.complete({
            "output_json": str(output_path),
            "frames": len(frames),
            "annotations": len(coco["annotations"]),
            "tracks": len(track_ids),
        })
    except Exception as e:
        logger.exception(f"Tracking job {job_id} failed")
        lifecycle.fail(str(e))
    finally:
        manager.release("sam3_video")
        cancel_flags.pop(job_id, None)
