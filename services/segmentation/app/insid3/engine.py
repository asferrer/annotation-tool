"""
INSID3 propagation job: 1 annotated example -> segment a whole image directory.

For each target image the reference (image + rasterized polygon mask) is
re-set (INSID3.segment() clears its state after every call), the predicted
mask is converted to COCO polygons and appended to the output COCO file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from app.video.tracker import mask_to_polygons
from .schemas import PropagateRequest

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

cancel_flags: Dict[str, bool] = {}


def _list_targets(request: PropagateRequest) -> List[Path]:
    root = Path(request.target_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Target directory not found: {root}")
    ref = Path(request.reference_image_path).resolve()
    targets = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and (request.include_reference or p.resolve() != ref)
    )
    if request.max_images:
        targets = targets[: request.max_images]
    if not targets:
        raise FileNotFoundError(f"No images found in {root}")
    return targets


def _reference_mask(request: PropagateRequest) -> Image.Image:
    """Rasterize the reference COCO polygons at the reference image size."""
    ref = cv2.imread(request.reference_image_path)
    if ref is None:
        raise FileNotFoundError(f"Reference image not readable: {request.reference_image_path}")
    mask = np.zeros(ref.shape[:2], dtype=np.uint8)
    for poly in request.reference_polygon:
        points = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    if mask.sum() == 0:
        raise ValueError("Reference polygon rasterized to an empty mask")
    return Image.fromarray(mask)


def _base_coco(request: PropagateRequest) -> Dict:
    return {
        "info": {"description": "INSID3 one-shot propagation", "source": "annotation-tool"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": request.label}],
    }


def run_propagation_job(job_id: str, request: PropagateRequest, lifecycle) -> None:
    """Worker-thread entrypoint."""
    from app.model_manager import manager

    cancel_flags.pop(job_id, None)
    targets = _list_targets(request)
    ref_image = Image.open(request.reference_image_path).convert("RGB")
    ref_mask = _reference_mask(request)

    coco = _base_coco(request)
    output_path = Path(request.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = manager.acquire("insid3")
    try:
        model.merge_threshold = request.merge_threshold
        lifecycle.start(f"Propagating '{request.label}' to {len(targets)} images")

        found = 0
        for index, target in enumerate(targets):
            if cancel_flags.get(job_id):
                lifecycle.cancel({"processed_images": index, "objects_found": found})
                return

            found += _propagate_one(model, ref_image, ref_mask, target, index, request, coco)

            if (index + 1) % CHECKPOINT_EVERY == 0 or index == len(targets) - 1:
                output_path.write_text(json.dumps(coco))
                lifecycle.update_progress(
                    index + 1, current_item=target.name,
                    details={"objects_found": found},
                )

        output_path.write_text(json.dumps(coco, indent=2))
        lifecycle.db.update_job_output(job_id, str(output_path))
        lifecycle.complete({
            "output_json": str(output_path),
            "images": len(targets),
            "objects_found": found,
        })
    except Exception as e:
        logger.exception(f"Propagation job {job_id} failed")
        lifecycle.fail(str(e))
    finally:
        manager.release("insid3")
        cancel_flags.pop(job_id, None)


def _propagate_one(
    model, ref_image: Image.Image, ref_mask: Image.Image,
    target: Path, index: int, request: PropagateRequest, coco: Dict,
) -> int:
    """Segment one target image and append results to the COCO dict. Returns 1 if found."""
    tgt_image = Image.open(target).convert("RGB")
    width, height = tgt_image.size

    # segment() resets model state, so the reference is re-set per target
    model.set_reference(ref_image, ref_mask)
    model.set_target(tgt_image)
    pred = model.segment()

    image_id = index + 1
    coco["images"].append({
        "id": image_id, "file_name": str(target), "width": width, "height": height,
    })

    mask = pred.cpu().numpy().astype(bool)
    if mask.sum() < request.min_area:
        return 0
    polygons = mask_to_polygons(mask)
    if not polygons:
        return 0

    ys, xs = np.where(mask)
    x0, y0 = float(xs.min()), float(ys.min())
    bbox = [x0, y0, float(xs.max()) - x0, float(ys.max()) - y0]
    coco["annotations"].append({
        "id": len(coco["annotations"]) + 1,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [round(v, 2) for v in bbox],
        "area": int(mask.sum()),
        "iscrowd": 0,
        "segmentation": polygons,
        "source": "insid3",
    })
    return 1
