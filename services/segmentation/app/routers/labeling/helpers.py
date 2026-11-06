"""
Shared helpers for labeling jobs: preview drawing, checkpoints, cooldown and format export.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np

try:
    from shared.legacy_export import export_yolo as shared_export_yolo, export_voc as shared_export_voc
except ImportError:
    import sys
    sys.path.insert(0, '/app')
    from shared.legacy_export import export_yolo as shared_export_yolo, export_voc as shared_export_voc

logger = logging.getLogger(__name__)


def _draw_annotations_on_image(
    image: np.ndarray,
    annotations: List[Dict],
    category_map: Dict[int, str],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True
) -> np.ndarray:
    """Draw annotations (bboxes, masks, labels) on an image for visualization.

    Args:
        image: BGR image array
        annotations: List of COCO-format annotations for this image
        category_map: Dict mapping category_id to category name
        draw_masks: Whether to draw segmentation masks
        draw_boxes: Whether to draw bounding boxes
        draw_labels: Whether to draw class labels

    Returns:
        Annotated image (BGR)
    """
    vis_image = image.copy()
    h, w = vis_image.shape[:2]

    # Generate distinct colors for each category
    np.random.seed(42)  # For consistent colors
    colors = {}
    for cat_id in category_map.keys():
        colors[cat_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))

    # Draw all masks with a single overlay operation (O(1) copies instead of O(N))
    if draw_masks:
        overlay = vis_image.copy()
        for ann in annotations:
            cat_id = ann.get("category_id", 1)
            color = colors.get(cat_id, (0, 255, 0))
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
        del overlay

    # Draw bounding boxes and labels
    for ann in annotations:
        cat_id = ann.get("category_id", 1)
        cat_name = category_map.get(cat_id, f"class_{cat_id}")
        color = colors.get(cat_id, (0, 255, 0))

        if draw_boxes and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)

        if draw_labels and "bbox" in ann:
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            label = f"{cat_name}"
            if "score" in ann:
                label += f" {ann['score']:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x, y - text_h - 4), (x + text_w + 4, y), color, -1)
            cv2.putText(vis_image, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def _save_preview_image(
    output_dir: Path,
    image: np.ndarray,
    image_name: str,
    annotations: List[Dict],
    category_map: Dict[int, str],
    preview_idx: int
) -> Optional[str]:
    """Save a preview image with annotations drawn.

    Returns the relative path to the saved preview, or None if failed.
    """
    try:
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        # Draw annotations
        vis_image = _draw_annotations_on_image(
            image, annotations, category_map,
            draw_masks=True, draw_boxes=True, draw_labels=True
        )

        # Save with a consistent naming scheme
        preview_name = f"preview_{preview_idx:04d}_{Path(image_name).stem}.jpg"
        preview_path = previews_dir / preview_name

        # Resize if too large (max 1024px on longest side) for faster loading
        h, w = vis_image.shape[:2]
        max_size = 1024
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            vis_image = cv2.resize(vis_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(preview_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(preview_path)
    except Exception as e:
        logger.warning(f"Failed to save preview image: {e}")
        return None


def _get_image_paths_from_output_dir(output_dir: Path) -> List[str]:
    """Reconstruct image paths from checkpoint/coco file in output directory.

    This is used when resuming a job after service restart.
    """
    image_paths = []

    # Try to load from annotations.json or checkpoint
    coco_path = output_dir / "annotations.json"
    checkpoint_path = output_dir / "checkpoint.json"

    coco_data = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_data = checkpoint.get("coco_data")
        except Exception:
            pass

    if coco_data is None and coco_path.exists():
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
        except Exception:
            pass

    if coco_data and "images" in coco_data:
        # Get image directory from first image path or use output_dir parent
        images_dir = output_dir.parent / "images"
        if not images_dir.exists():
            images_dir = output_dir.parent

        for img_info in coco_data["images"]:
            file_name = img_info.get("file_name", "")
            # Try to find the image file
            for search_dir in [images_dir, output_dir.parent, output_dir]:
                potential_path = search_dir / file_name
                if potential_path.exists():
                    image_paths.append(str(potential_path))
                    break

    return image_paths


def _apply_padding_to_bbox(bbox: List[float], padding: int, img_width: int, img_height: int) -> List[float]:
    """Apply padding to a bounding box while keeping it within image bounds.

    Args:
        bbox: [x, y, width, height] format bounding box
        padding: Pixels of padding to add on each side
        img_width: Image width for bounds checking
        img_height: Image height for bounds checking

    Returns:
        Padded bbox [x, y, width, height] clamped to image bounds
    """
    if padding <= 0:
        return bbox

    x, y, w, h = bbox
    # Expand the bbox by padding on each side
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_width, x + w + padding)
    y2 = min(img_height, y + h + padding)

    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


async def _thermal_cooldown(cooldown_s: float, vram_monitor) -> None:
    """Sleep for cooldown_s, then add extra delay if GPU temperature is too high.

    Thresholds are read from environment variables:
      GPU_TEMP_THROTTLE_C  (default 80): start adding extra sleep
      GPU_TEMP_MAX_C       (default 87): max extra sleep (10 s)

    Extra sleep scales linearly from 2 s at throttle temp to 10 s at max temp.
    """
    await asyncio.sleep(cooldown_s)

    if vram_monitor is None:
        return

    temp = vram_monitor.get_temperature()
    if temp is None:
        return

    temp_throttle = int(os.environ.get("GPU_TEMP_THROTTLE_C", "80"))
    temp_max = int(os.environ.get("GPU_TEMP_MAX_C", "87"))

    if temp >= temp_throttle:
        ratio = min(1.0, (temp - temp_throttle) / max(1, temp_max - temp_throttle))
        extra = 2.0 + ratio * 8.0  # 2 s → 10 s
        logger.info(f"GPU temp {temp}°C (throttle={temp_throttle}°C) — extra cooldown {extra:.1f}s")
        await asyncio.sleep(extra)


def _save_labeling_checkpoint(checkpoint_path: Path, coco_data: Dict, next_annotation_id: int,
                               last_processed_idx: int, objects_by_class: Dict[str, int]):
    """Save checkpoint for labeling job recovery."""
    checkpoint = {
        "coco_data": coco_data,
        "next_annotation_id": next_annotation_id,
        "last_processed_idx": last_processed_idx,
        "objects_by_class": objects_by_class,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def _export_to_yolo(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to YOLO format (shared bbox-only exporter)."""
    shared_export_yolo(coco_data, output_dir)


def _export_to_voc(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to Pascal VOC format (shared bbox-only exporter)."""
    shared_export_voc(coco_data, output_dir)
