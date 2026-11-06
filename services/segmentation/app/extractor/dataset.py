"""
Batch extraction from a COCO dataset.
"""

import logging
import json
import asyncio
import gc
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Callable
from functools import partial

import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.models.extraction_schemas import ExtractionSummary

if TYPE_CHECKING:
    from app.models.extraction_schemas import DeduplicationConfig

logger = logging.getLogger(__name__)


class DatasetMixin:
    """Batch extraction from a COCO dataset."""

    def _extract_from_dataset_sync(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_dir: str,
        categories_to_extract: List[str] = None,
        use_sam3_for_bbox: bool = True,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        deduplication_config: Optional['DeduplicationConfig'] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset with deduplication support.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates
            deduplication_config: Configuration for deduplication (None = default enabled)

        Returns:
            Extraction results dictionary with deduplication stats
        """
        start_time = datetime.now()

        # Build lookups
        categories = {c["id"]: c for c in coco_data.get("categories", [])}
        images = {i["id"]: i for i in coco_data.get("images", [])}
        annotations = coco_data.get("annotations", [])

        # Filter categories
        if categories_to_extract:
            category_ids = {
                cid for cid, cat in categories.items()
                if cat["name"] in categories_to_extract
            }
        else:
            category_ids = set(categories.keys())

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for cat in categories.values():
            if cat["id"] in category_ids:
                (output_path / cat["name"]).mkdir(exist_ok=True)

        # Process annotations
        results = {
            "extracted": 0,
            "failed": 0,
            "by_category": {categories[cid]["name"]: 0 for cid in category_ids},
            "by_method": {},
            "errors": [],
            "extracted_files": [],
            "deduplication_stats": {
                "duplicates_prevented": 0,
                "enabled": deduplication_config.enabled if deduplication_config else True
            }
        }

        # Group annotations differently based on mode
        if force_sam3_text_prompt:
            # Text prompt mode: group by (image_id, category_name) for batch processing
            annotations_grouped = {}
            for ann in annotations:
                if ann.get("category_id") not in category_ids:
                    continue

                # Check minimum area
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4 and bbox[2] * bbox[3] < min_object_area:
                    continue

                img_id = ann.get("image_id")
                cat_name = categories[ann["category_id"]]["name"]
                key = (img_id, cat_name)

                if key not in annotations_grouped:
                    annotations_grouped[key] = []
                annotations_grouped[key].append(ann)
        else:
            # Bbox/mask mode: group by image_id only
            annotations_grouped = {}
            for ann in annotations:
                if ann.get("category_id") not in category_ids:
                    continue

                # Check minimum area
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4 and bbox[2] * bbox[3] < min_object_area:
                    continue

                img_id = ann.get("image_id")
                if img_id not in annotations_grouped:
                    annotations_grouped[img_id] = []
                annotations_grouped[img_id].append(ann)

        # Process each image
        current_image = None
        current_image_id = None
        registry = None

        # Get unique image IDs
        if force_sam3_text_prompt:
            unique_img_ids = sorted(set(key[0] for key in annotations_grouped.keys()))
        else:
            unique_img_ids = sorted(annotations_grouped.keys())

        for img_index, img_id in enumerate(unique_img_ids):
            # Load image once per image_id (reuse current_image)
            if current_image_id != img_id:
                current_image_id = img_id

                if img_id not in images:
                    # Fail all annotations for this image
                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(f"Image ID {img_id} not found in dataset")
                    continue

                img_info = images[img_id]
                img_path = self.resolve_image_path(images_dir, img_info)

                # Load image
                if not img_path.exists():
                    error_msg = f"Image not found: {img_path} (file_name in JSON: {img_info.get('file_name', '?')}, images_dir: {images_dir})"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                try:
                    current_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    error_msg = f"Exception loading image {img_path}: {e}"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                if current_image is None:
                    error_msg = f"cv2.imread returned None for: {img_path}"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                logger.debug(f"Loaded image {img_path}: shape={current_image.shape}, dtype={current_image.dtype}")

                # Create registry for this image
                if deduplication_config is None:
                    # Default: enabled with IoU=0.7
                    from app.models.extraction_schemas import DeduplicationConfig
                    deduplication_config = DeduplicationConfig(enabled=True, iou_threshold=0.7)

                if deduplication_config.enabled:
                    from app.utils import ExtractionRegistry
                    registry = ExtractionRegistry(
                        iou_threshold=deduplication_config.iou_threshold,
                        cross_category_dedup=deduplication_config.cross_category_dedup,
                        matching_strategy=deduplication_config.matching_strategy
                    )
                    logger.info(
                        f"Created ExtractionRegistry for image {img_id} "
                        f"(iou_threshold={deduplication_config.iou_threshold}, "
                        f"cross_category={deduplication_config.cross_category_dedup})"
                    )
                else:
                    registry = None

            # Process based on mode
            if force_sam3_text_prompt:
                self._process_text_prompt_mode(
                    image=current_image,
                    img_id=img_id,
                    img_info=images[img_id],
                    annotations_grouped=annotations_grouped,
                    registry=registry,
                    results=results,
                    categories=categories,
                    output_path=output_path,
                    padding=padding,
                    min_object_area=min_object_area,
                    save_individual_coco=save_individual_coco,
                    progress_callback=progress_callback
                )
            else:
                self._process_bbox_mask_mode(
                    image=current_image,
                    img_id=img_id,
                    img_info=images[img_id],
                    annotations_grouped=annotations_grouped,
                    registry=registry,
                    results=results,
                    categories=categories,
                    output_path=output_path,
                    use_sam3_for_bbox=use_sam3_for_bbox,
                    force_bbox_only=force_bbox_only,
                    force_sam3_resegmentation=force_sam3_resegmentation,
                    padding=padding,
                    save_individual_coco=save_individual_coco,
                    progress_callback=progress_callback
                )

            # Collect dedup stats from registry
            if registry:
                stats = registry.get_stats()
                results["deduplication_stats"]["duplicates_prevented"] += stats["duplicates_prevented"]

            # Memory cleanup after processing each image to prevent memory buildup
            # Release the image reference and run garbage collection periodically
            if current_image is not None:
                del current_image
                current_image = None

            # Run garbage collection every few images to prevent memory buildup
            if (img_index + 1) % 5 == 0:  # Every 5 images
                gc.collect()
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final cleanup
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Save extraction summary
        summary = ExtractionSummary(
            extraction_date=start_time.isoformat(),
            source_dataset=coco_data.get("info", {}).get("description", "Unknown"),
            images_dir=images_dir,
            output_dir=output_dir,
            total_objects_extracted=results["extracted"],
            categories={
                name: {
                    "count": count,
                    "method": "mixed" if len(results["by_method"]) > 1 else list(results["by_method"].keys())[0] if results["by_method"] else "none"
                }
                for name, count in results["by_category"].items()
                if count > 0
            },
            failed_extractions=results["failed"],
            errors=results["errors"][:100],  # Limit errors in summary
            processing_time_seconds=processing_time
        )

        summary_path = output_path / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.model_dump(), f, indent=2)

        results["summary_path"] = str(summary_path)
        results["processing_time_seconds"] = processing_time

        logger.info(
            f"Extraction complete: {results['extracted']} extracted, "
            f"{results['failed']} failed, "
            f"{results['deduplication_stats']['duplicates_prevented']} duplicates prevented"
        )

        return results

    async def extract_from_dataset(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_dir: str,
        categories_to_extract: List[str] = None,
        use_sam3_for_bbox: bool = True,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        deduplication_config: Optional['DeduplicationConfig'] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset with deduplication support.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates
            deduplication_config: Configuration for deduplication (None = default enabled)

        Returns:
            Extraction results dictionary with deduplication stats
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_from_dataset_sync,
                coco_data=coco_data,
                images_dir=images_dir,
                output_dir=output_dir,
                categories_to_extract=categories_to_extract,
                use_sam3_for_bbox=use_sam3_for_bbox,
                force_bbox_only=force_bbox_only,
                force_sam3_resegmentation=force_sam3_resegmentation,
                force_sam3_text_prompt=force_sam3_text_prompt,
                padding=padding,
                min_object_area=min_object_area,
                save_individual_coco=save_individual_coco,
                progress_callback=progress_callback,
                deduplication_config=deduplication_config
            )
        )
