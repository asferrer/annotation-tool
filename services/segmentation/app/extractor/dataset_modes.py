"""
Per-annotation processing modes used by the batch extraction (text prompt and bbox/mask).
"""

import logging
import json
import base64
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Callable

import cv2
import numpy as np

from app.models.extraction_schemas import ExtractionMethod

if TYPE_CHECKING:
    from app.utils import ExtractionRegistry

logger = logging.getLogger(__name__)


class DatasetModesMixin:
    """Per-annotation processing modes used by the batch extraction (text prompt and bbox/mask)."""

    def _process_text_prompt_mode(
        self,
        image: np.ndarray,
        img_id: int,
        img_info: Dict,
        annotations_grouped: Dict,
        registry: Optional['ExtractionRegistry'],
        results: Dict,
        categories: Dict,
        output_path: Path,
        padding: int,
        min_object_area: int,
        save_individual_coco: bool,
        progress_callback: Callable
    ) -> None:
        """
        Process annotations in text prompt mode with instance matching.

        Algorithm:
        1. Group annotations by category for this image
        2. For each category:
           - Check registry cache for SAM3 results
           - If not cached: Run SAM3 once → get N instances → cache
           - Match N instances to M annotations (one-to-one)
           - For each matched pair: check dedup, extract, register
           - For unmatched: FAIL with clear message

        Args:
            image: Loaded image
            img_id: Image ID
            img_info: Image metadata from COCO JSON
            annotations_grouped: Dict with (img_id, cat_name) keys
            registry: ExtractionRegistry or None
            results: Results dict to update
            categories: Category lookup dict
            output_path: Output directory Path
            padding: Padding pixels
            min_object_area: Minimum area threshold
            save_individual_coco: Whether to save individual JSON
            progress_callback: Progress callback function
        """
        from app.utils import match_instances_to_annotations, MatchingStrategy

        # Get all annotations for this image, grouped by category
        anns_by_category = {}
        for key, anns in annotations_grouped.items():
            if isinstance(key, tuple) and key[0] == img_id:
                cat_name = key[1]
                anns_by_category[cat_name] = anns

        # Process each category
        for cat_name, annotations in anns_by_category.items():
            logger.info(f"Processing category '{cat_name}': {len(annotations)} annotations (text prompt mode)")

            # Check cache
            instances = registry.get_sam3_instances(cat_name) if registry else None

            if instances is None:
                # Run SAM3 ONCE for this category
                instances = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=cat_name,
                    min_area=min_object_area,
                    return_all_instances=True  # Get ALL instances
                )

                if instances is None or len(instances) == 0:
                    # SAM3 failed - fail all annotations
                    logger.warning(f"SAM3 text prompt failed for '{cat_name}' - marking {len(annotations)} annotations as failed")
                    for ann in annotations:
                        results["failed"] += 1
                        results["errors"].append(
                            f"SAM3 text prompt returned no instances for '{cat_name}' (ann_id={ann['id']})"
                        )
                    continue

                # Cache results
                if registry:
                    registry.cache_sam3_results(cat_name, instances)

                logger.info(f"SAM3 found {len(instances)} instances for '{cat_name}' ({len(annotations)} annotations)")
            else:
                logger.debug(f"Using cached SAM3 results for '{cat_name}': {len(instances)} instances")

            # Match instances to annotations
            matching_strategy = (registry.matching_strategy
                               if registry
                               else MatchingStrategy.BBOX_IOU)

            assignment_map = match_instances_to_annotations(
                instances=instances,
                annotations=annotations,
                strategy=matching_strategy
            )

            matched_count = sum(1 for v in assignment_map.values() if v is not None)
            logger.info(f"Matched {matched_count}/{len(annotations)} annotations to instances")

            # Extract each matched annotation
            for ann in annotations:
                ann_id = ann["id"]
                cat_id = ann["category_id"]
                instance_idx = assignment_map.get(ann_id)

                if instance_idx is None:
                    # No matching instance - FAIL
                    results["failed"] += 1
                    results["errors"].append(
                        f"No SAM3 instance matched annotation {ann_id} "
                        f"(class='{cat_name}', {len(instances)} instances available)"
                    )
                    logger.debug(f"Annotation {ann_id} unmatched (bbox={ann.get('bbox')})")
                    continue

                instance = instances[instance_idx]

                # Check for duplicates
                if registry:
                    is_dup, dup_id, max_iou = registry.is_duplicate(
                        mask=instance.mask,
                        bbox=instance.bbox,
                        category_id=cat_id,
                        category_name=cat_name
                    )

                    if is_dup:
                        logger.info(
                            f"Skipping duplicate: ann_id={ann_id} (IoU={max_iou:.3f} "
                            f"with ann_id={dup_id}, class='{cat_name}')"
                        )
                        results["failed"] += 1
                        results["errors"].append(
                            f"Duplicate of annotation {dup_id} (IoU={max_iou:.3f})"
                        )
                        continue

                # Extract object with this mask
                try:
                    cropped, new_bbox = self.crop_with_mask(
                        image, instance.mask, instance.bbox, padding
                    )

                    if cropped is None:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to crop annotation {ann_id}")
                        continue

                    # Encode to base64
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to encode annotation {ann_id}")
                        continue

                    # Save PNG
                    png_path = output_path / cat_name / f"{ann_id}.png"
                    with open(png_path, 'wb') as f:
                        f.write(encoded.tobytes())

                    # Save JSON if requested
                    json_path = None
                    if save_individual_coco:
                        json_path = output_path / cat_name / f"{ann_id}.json"
                        cat = categories[cat_id]
                        coco_json = self.create_individual_coco(
                            annotation=ann,
                            category=cat,
                            image_shape=cropped.shape[:2],
                            original_image_filename=f"image_{img_id}"
                        )
                        with open(json_path, 'w') as f:
                            json.dump(coco_json, f, indent=2)

                    # Register extraction
                    if registry:
                        registry.register_extraction(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            annotation_id=ann_id,
                            category_id=cat_id,
                            category_name=cat_name,
                            method=ExtractionMethod.SAM3_TEXT_PROMPT.value
                        )

                    # Update results
                    results["extracted"] += 1
                    results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1
                    results["by_method"][ExtractionMethod.SAM3_TEXT_PROMPT.value] = \
                        results["by_method"].get(ExtractionMethod.SAM3_TEXT_PROMPT.value, 0) + 1

                    results["extracted_files"].append({
                        "annotation_id": ann_id,
                        "category_name": cat_name,
                        "image_path": str(png_path),
                        "json_path": str(json_path) if json_path else None,
                        "method": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                        "original_bbox": instance.bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]]
                    })

                    # Progress callback
                    if progress_callback:
                        progress_callback({
                            "extracted": results["extracted"],
                            "failed": results["failed"],
                            "current_category": cat_name,
                            "by_category": results["by_category"]
                        })

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Annotation {ann_id}: {str(e)}")
                    logger.error(f"Error extracting annotation {ann_id}: {e}")

    def _process_bbox_mask_mode(
        self,
        image: np.ndarray,
        img_id: int,
        img_info: Dict,
        annotations_grouped: Dict,
        registry: Optional['ExtractionRegistry'],
        results: Dict,
        categories: Dict,
        output_path: Path,
        use_sam3_for_bbox: bool,
        force_bbox_only: bool,
        force_sam3_resegmentation: bool,
        padding: int,
        save_individual_coco: bool,
        progress_callback: Callable
    ) -> None:
        """
        Process annotations in bbox/mask mode with optional deduplication.

        Algorithm:
        1. For each annotation:
           - Call _extract_single_object_sync (existing logic)
           - If dedup enabled: check IoU with registry
           - Save if unique, register

        Args:
            image: Loaded image
            img_info: Image metadata from COCO JSON
            img_id: Image ID
            annotations_grouped: Dict with img_id keys
            registry: ExtractionRegistry or None
            results: Results dict to update
            categories: Category lookup dict
            output_path: Output directory Path
            use_sam3_for_bbox: Use SAM3 for bbox-only
            force_bbox_only: Force bbox-only mode
            force_sam3_resegmentation: Force SAM3 reseg
            padding: Padding pixels
            save_individual_coco: Whether to save individual JSON
            progress_callback: Progress callback function
        """
        # Get annotations for this image
        annotations = annotations_grouped.get(img_id, [])

        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = categories[cat_id]["name"]
            ann_id = ann["id"]

            try:
                # Extract using existing logic
                extraction_result = self._extract_single_object_sync(
                    image=image,
                    annotation=ann,
                    category_name=cat_name,
                    use_sam3=use_sam3_for_bbox,
                    padding=padding,
                    force_bbox_only=force_bbox_only,
                    force_sam3_resegmentation=force_sam3_resegmentation,
                    force_sam3_text_prompt=False
                )

                if not extraction_result["success"]:
                    results["failed"] += 1
                    results["errors"].append(extraction_result.get("error", "Unknown error"))
                    continue

                # Deduplication check
                if registry and extraction_result.get("mask_base64"):
                    # Decode mask from base64
                    mask_data = base64.b64decode(extraction_result["mask_base64"])
                    mask_img = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)

                    bbox = extraction_result["original_bbox"]

                    is_dup, dup_id, max_iou = registry.is_duplicate(
                        mask=mask_img,
                        bbox=bbox,
                        category_id=cat_id,
                        category_name=cat_name
                    )

                    if is_dup:
                        logger.info(
                            f"Skipping duplicate: ann_id={ann_id} (IoU={max_iou:.3f} "
                            f"with ann_id={dup_id}, class='{cat_name}')"
                        )
                        results["failed"] += 1
                        results["errors"].append(
                            f"Duplicate of annotation {dup_id} (IoU={max_iou:.3f})"
                        )
                        continue

                # Save PNG
                png_path = output_path / cat_name / f"{ann_id}.png"
                img_data = base64.b64decode(extraction_result["cropped_image_base64"])
                with open(png_path, 'wb') as f:
                    f.write(img_data)

                # Save JSON
                json_path = None
                if save_individual_coco:
                    json_path = output_path / cat_name / f"{ann_id}.json"
                    cat = categories[cat_id]
                    coco_json = self.create_individual_coco(
                        annotation=ann,
                        category=cat,
                        image_shape=extraction_result["extracted_size"][::-1],
                        original_image_filename=f"image_{img_id}"
                    )
                    with open(json_path, 'w') as f:
                        json.dump(coco_json, f, indent=2)

                # Register
                if registry and extraction_result.get("mask_base64"):
                    registry.register_extraction(
                        mask=mask_img,
                        bbox=bbox,
                        annotation_id=ann_id,
                        category_id=cat_id,
                        category_name=cat_name,
                        method=extraction_result["method_used"]
                    )

                # Update results
                results["extracted"] += 1
                results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1
                method = extraction_result["method_used"]
                results["by_method"][method] = results["by_method"].get(method, 0) + 1

                results["extracted_files"].append({
                    "annotation_id": ann_id,
                    "category_name": cat_name,
                    "image_path": str(png_path),
                    "json_path": str(json_path) if json_path else None,
                    "method": method,
                    "original_bbox": extraction_result["original_bbox"],
                    "extracted_size": extraction_result["extracted_size"]
                })

                # Progress
                if progress_callback:
                    progress_callback({
                        "extracted": results["extracted"],
                        "failed": results["failed"],
                        "current_category": cat_name,
                        "by_category": results["by_category"]
                    })

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Annotation {ann_id}: {str(e)}")
                logger.error(f"Error extracting annotation {ann_id}: {e}")
