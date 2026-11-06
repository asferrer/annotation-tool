"""
Object cropping, per-object COCO files, dataset analysis and image-path resolution.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

from app.models.extraction_schemas import AnnotationType

logger = logging.getLogger(__name__)


class CropMixin:
    """Object cropping, per-object COCO files, dataset analysis and image-path resolution."""

    @staticmethod
    def crop_with_mask(
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[float],
        padding: int = 5
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Crop image using mask and add alpha channel.

        Args:
            image: Source image (BGR)
            mask: Binary mask (0 or 255)
            bbox: Bounding box [x, y, width, height]
            padding: Pixels of padding around object

        Returns:
            Tuple of (RGBA cropped image, new bbox [x, y, w, h]) or (None, []) on error
        """
        if image is None or image.size == 0:
            logger.error("crop_with_mask: received empty or None image")
            return None, []

        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]

        # Compute intersection of bbox with image bounds
        # bbox region: [x, y] to [x+bw, y+bh]
        # image region: [0, 0] to [w, h]
        x1_bbox, y1_bbox = x, y
        x2_bbox, y2_bbox = x + bw, y + bh

        # Intersection
        x1_valid = max(0, x1_bbox)
        y1_valid = max(0, y1_bbox)
        x2_valid = min(w, x2_bbox)
        y2_valid = min(h, y2_bbox)

        # Check if there's a valid intersection (bbox overlaps with image)
        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            logger.info(f"crop_with_mask: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping annotation (data quality issue).")
            return None, []

        # Apply padding around the valid region, clamped to image bounds
        x1 = max(0, x1_valid - padding)
        y1 = max(0, y1_valid - padding)
        x2 = min(w, x2_valid + padding)
        y2 = min(h, y2_valid + padding)

        # Final validation (should always pass after above logic)
        if x1 >= x2 or y1 >= y2:
            logger.error(f"crop_with_mask: invalid crop after padding x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None, []

        # Crop image and mask
        cropped = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()

        # Validate cropped image
        if cropped is None or cropped.size == 0:
            logger.error(f"crop_with_mask: cropped image is empty for bbox={bbox}")
            return None, []

        # Convert to RGBA
        try:
            if len(cropped.shape) == 2:
                # Grayscale
                rgba = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGRA)
            elif cropped.shape[2] == 3:
                # BGR
                rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            else:
                # Already has alpha
                rgba = cropped.copy()
        except cv2.error as e:
            logger.error(f"crop_with_mask: cvtColor failed: {e}, cropped shape={cropped.shape}")
            return None, []

        # Apply mask to alpha channel
        rgba[:, :, 3] = cropped_mask

        return rgba, [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def crop_bbox_only(
        image: np.ndarray,
        bbox: List[float],
        padding: int = 5
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Crop image using only bounding box (no transparency).

        Args:
            image: Source image
            bbox: Bounding box [x, y, width, height]
            padding: Pixels of padding

        Returns:
            Tuple of (cropped image, new bbox) or (None, []) on error
        """
        if image is None or image.size == 0:
            logger.error("crop_bbox_only: received empty or None image")
            return None, []

        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]

        # Compute intersection of bbox with image bounds
        # bbox region: [x, y] to [x+bw, y+bh]
        # image region: [0, 0] to [w, h]
        x1_bbox, y1_bbox = x, y
        x2_bbox, y2_bbox = x + bw, y + bh

        # Intersection
        x1_valid = max(0, x1_bbox)
        y1_valid = max(0, y1_bbox)
        x2_valid = min(w, x2_bbox)
        y2_valid = min(h, y2_bbox)

        # Check if there's a valid intersection (bbox overlaps with image)
        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            logger.info(f"crop_bbox_only: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping annotation (data quality issue).")
            return None, []

        # Apply padding around the valid region, clamped to image bounds
        x1 = max(0, x1_valid - padding)
        y1 = max(0, y1_valid - padding)
        x2 = min(w, x2_valid + padding)
        y2 = min(h, y2_valid + padding)

        # Final validation (should always pass after above logic)
        if x1 >= x2 or y1 >= y2:
            logger.error(f"crop_bbox_only: invalid crop after padding x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None, []

        cropped = image[y1:y2, x1:x2].copy()

        # Validate cropped image
        if cropped is None or cropped.size == 0:
            logger.error(f"crop_bbox_only: cropped image is empty for bbox={bbox}")
            return None, []

        # Convert to RGBA with full opacity
        try:
            if len(cropped.shape) == 2:
                rgba = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGRA)
            elif cropped.shape[2] == 3:
                rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            else:
                rgba = cropped.copy()
        except cv2.error as e:
            logger.error(f"crop_bbox_only: cvtColor failed: {e}, cropped shape={cropped.shape}")
            return None, []

        return rgba, [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def create_individual_coco(
        annotation: Dict[str, Any],
        category: Dict[str, Any],
        image_shape: Tuple[int, int],
        original_image_filename: str
    ) -> Dict[str, Any]:
        """
        Create individual COCO JSON for an extracted object.

        Args:
            annotation: Original COCO annotation
            category: Category information
            image_shape: Shape of extracted image (height, width)
            original_image_filename: Original source image filename

        Returns:
            COCO format dictionary for the individual object
        """
        h, w = image_shape[:2]

        return {
            "info": {
                "description": "Extracted object from COCO dataset",
                "version": "1.0",
                "extracted_from": original_image_filename,
                "original_annotation_id": annotation.get("id"),
                "extraction_date": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [{
                "id": 1,
                "file_name": f"{annotation['id']}.png",
                "width": w,
                "height": h
            }],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [[0, 0, w, 0, w, h, 0, h]]  # Full image polygon
            }],
            "categories": [{
                "id": 1,
                "name": category.get("name", "object"),
                "supercategory": category.get("supercategory", "none")
            }]
        }

    def analyze_dataset(
        self,
        coco_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a COCO dataset to determine annotation types.

        Args:
            coco_data: COCO format dataset dictionary

        Returns:
            Analysis results dictionary
        """
        categories = {c["id"]: c for c in coco_data.get("categories", [])}
        annotations = coco_data.get("annotations", [])
        images = coco_data.get("images", [])

        # Count by category and type
        category_stats = {}
        for cat_id, cat in categories.items():
            category_stats[cat_id] = {
                "id": cat_id,
                "name": cat["name"],
                "count": 0,
                "with_segmentation": 0,
                "bbox_only": 0
            }

        total_with_seg = 0
        total_bbox_only = 0
        sample_annotation = None

        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id not in category_stats:
                continue

            ann_type = self.detect_annotation_type(ann)
            category_stats[cat_id]["count"] += 1

            if ann_type in [AnnotationType.POLYGON, AnnotationType.RLE]:
                category_stats[cat_id]["with_segmentation"] += 1
                total_with_seg += 1
            else:
                category_stats[cat_id]["bbox_only"] += 1
                total_bbox_only += 1

            if sample_annotation is None:
                sample_annotation = ann

        # Determine recommendation
        if total_bbox_only == 0:
            recommendation = "use_masks"
        elif total_with_seg == 0:
            recommendation = "use_sam3"
        else:
            recommendation = "mixed"

        return {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "annotations_with_segmentation": total_with_seg,
            "annotations_bbox_only": total_bbox_only,
            "categories": list(category_stats.values()),
            "recommendation": recommendation,
            "sample_annotation": sample_annotation
        }

    @staticmethod
    def resolve_image_path(images_base_dir: str, coco_image: Dict[str, Any]) -> Path:
        """
        Resolve full path to an image.

        Args:
            images_base_dir: Base directory for images
            coco_image: COCO image entry with file_name

        Returns:
            Full path to image file
        """
        file_name = coco_image.get("file_name", "")

        # Handle empty file_name
        if not file_name:
            logger.warning(f"Empty file_name in COCO image entry: {coco_image}")
            return Path("")

        # Normalize path separators (handle Windows vs Unix)
        file_name = file_name.replace("\\", "/")

        # If absolute path, use directly
        if os.path.isabs(file_name):
            return Path(file_name)

        # Combine with base directory
        base_path = Path(images_base_dir)
        full_path = base_path / file_name

        # Log for debugging
        if not full_path.exists():
            logger.warning(f"Image path does not exist: {full_path} (base: {images_base_dir}, file_name: {file_name})")

        return full_path
