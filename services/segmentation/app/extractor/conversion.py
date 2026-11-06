"""
Dataset conversion from bounding boxes to segmentation masks.
"""

import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable
from functools import partial

import cv2
import numpy as np

from app.models.extraction_schemas import AnnotationType

logger = logging.getLogger(__name__)


class ConversionMixin:
    """Dataset conversion from bounding boxes to segmentation masks."""

    def _convert_bbox_to_segmentation_sync(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_path: str,
        categories_to_convert: List[str] = None,
        overwrite_existing: bool = False,
        simplify_polygons: bool = True,
        simplify_tolerance: float = 2.0,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
    ) -> Dict[str, Any]:
        """
        Convert bbox-only annotations to segmentations using SAM3.

        Args:
            coco_data: Input COCO dataset
            images_dir: Directory containing images
            output_path: Path for output COCO JSON
            categories_to_convert: Categories to convert (None = all)
            overwrite_existing: Overwrite existing segmentations
            simplify_polygons: Simplify generated polygons
            simplify_tolerance: Simplification tolerance
            progress_callback: Progress callback function

        Returns:
            Conversion results
        """
        if not self.sam3_available:
            return {
                "success": False,
                "error": "SAM3 not available for conversion"
            }

        start_time = datetime.now()

        # Deep copy to avoid modifying original
        import copy
        output_data = copy.deepcopy(coco_data)

        # Build lookups
        categories = {c["id"]: c for c in output_data.get("categories", [])}
        images = {i["id"]: i for i in output_data.get("images", [])}

        # Filter categories
        if categories_to_convert:
            category_ids = {
                cid for cid, cat in categories.items()
                if cat["name"] in categories_to_convert
            }
        else:
            category_ids = set(categories.keys())

        results = {
            "converted": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "by_category": {}
        }

        total = len(output_data.get("annotations", []))
        processed = 0

        # Group by image
        annotations_by_image = {}
        for idx, ann in enumerate(output_data["annotations"]):
            img_id = ann.get("image_id")
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append((idx, ann))

        for img_id, indexed_annotations in annotations_by_image.items():
            if img_id not in images:
                continue

            img_info = images[img_id]
            img_path = self.resolve_image_path(images_dir, img_info)

            if not img_path.exists():
                error_msg = f"Image not found: {img_path}"
                logger.error(error_msg)
                for idx, ann in indexed_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            image = cv2.imread(str(img_path))
            if image is None or image.size == 0:
                error_msg = f"cv2.imread failed for: {img_path}"
                logger.error(error_msg)
                for idx, ann in indexed_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            h, w = image.shape[:2]

            for idx, ann in indexed_annotations:
                processed += 1

                cat_id = ann.get("category_id")
                if cat_id not in category_ids:
                    results["skipped"] += 1
                    continue

                cat_name = categories[cat_id]["name"]
                ann_type = self.detect_annotation_type(ann)

                # Skip if already has segmentation and not overwriting
                if ann_type != AnnotationType.BBOX_ONLY and not overwrite_existing:
                    results["skipped"] += 1
                    continue

                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    results["skipped"] += 1
                    continue

                # Validate bbox dimensions
                bx, by, bw_val, bh_val = bbox[:4]
                if bw_val <= 0 or bh_val <= 0:
                    results["skipped"] += 1
                    results["errors"].append(f"Invalid bbox dimensions for annotation {ann.get('id')}: {bbox}")
                    continue

                # Check if bbox has overlap with image (intersection-based)
                x1_valid = max(0, int(bx))
                y1_valid = max(0, int(by))
                x2_valid = min(w, int(bx + bw_val))
                y2_valid = min(h, int(by + bh_val))

                if x1_valid >= x2_valid or y1_valid >= y2_valid:
                    results["skipped"] += 1
                    results["errors"].append(f"Annotation {ann.get('id')}: bbox outside image bounds - bbox=({bx},{by},{bw_val},{bh_val}) image=({w},{h})")
                    continue

                try:
                    # Segment with SAM3 (uses clamped bbox internally)
                    # Call sync version directly since we're in a thread
                    mask = self._segment_with_sam3_sync(image, bbox)

                    if mask is None:
                        results["failed"] += 1
                        results["errors"].append(f"SAM3 failed for annotation {ann.get('id')}")
                        continue

                    # Convert mask to polygon
                    polygons = self.mask_to_polygon(mask, simplify_polygons, simplify_tolerance)

                    if not polygons:
                        results["failed"] += 1
                        results["errors"].append(f"No valid polygon for annotation {ann.get('id')}")
                        continue

                    # Update annotation
                    output_data["annotations"][idx]["segmentation"] = polygons
                    output_data["annotations"][idx]["area"] = int(np.sum(mask > 0))

                    results["converted"] += 1
                    results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Error on annotation {ann.get('id')}: {str(e)}")

                if progress_callback:
                    progress_callback({
                        "converted": results["converted"],
                        "skipped": results["skipped"],
                        "failed": results["failed"],
                        "total": total,
                        "current_image": img_info.get("file_name", "")
                    })

        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        end_time = datetime.now()

        results["success"] = True
        results["output_path"] = str(output_file)
        results["processing_time_seconds"] = (end_time - start_time).total_seconds()

        return results

    async def convert_bbox_to_segmentation(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_path: str,
        categories_to_convert: List[str] = None,
        overwrite_existing: bool = False,
        simplify_polygons: bool = True,
        simplify_tolerance: float = 2.0,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
    ) -> Dict[str, Any]:
        """
        Convert bbox-only annotations to segmentations using SAM3.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            coco_data: Input COCO dataset
            images_dir: Directory containing images
            output_path: Path for output COCO JSON
            categories_to_convert: Categories to convert (None = all)
            overwrite_existing: Overwrite existing segmentations
            simplify_polygons: Simplify generated polygons
            simplify_tolerance: Simplification tolerance
            progress_callback: Progress callback function

        Returns:
            Conversion results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._convert_bbox_to_segmentation_sync,
                coco_data=coco_data,
                images_dir=images_dir,
                output_path=output_path,
                categories_to_convert=categories_to_convert,
                overwrite_existing=overwrite_existing,
                simplify_polygons=simplify_polygons,
                simplify_tolerance=simplify_tolerance,
                progress_callback=progress_callback,
            )
        )
