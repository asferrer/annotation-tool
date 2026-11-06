"""
Custom-object extraction driven by text prompts.
"""

import logging
import json
import asyncio
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from functools import partial

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class CustomObjectsMixin:
    """Custom-object extraction driven by text prompts."""

    def _extract_custom_objects_sync(
        self,
        images_dir: str,
        output_dir: str,
        object_names: List[str],
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication_config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Extract custom objects using text prompts only (no COCO JSON).
        Synchronous implementation for thread pool execution.

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for extracted objects
            object_names: List of object names to segment
            padding: Pixels of padding around objects
            min_object_area: Minimum object area in pixels
            save_individual_coco: Save individual COCO JSON per object
            deduplication_config: Deduplication configuration dict
            progress_callback: Progress callback function

        Returns:
            Extraction results dictionary
        """
        from app.utils import ExtractionRegistry
        from app.models.extraction_schemas import DeduplicationConfig

        start_time = datetime.now()

        # Parse deduplication config
        if deduplication_config:
            dedup_cfg = DeduplicationConfig(**deduplication_config)
        else:
            dedup_cfg = DeduplicationConfig(enabled=True, iou_threshold=0.7)

        # Initialize results
        results = {
            "success": False,
            "total_images": 0,
            "total_objects_extracted": 0,
            "failed_extractions": 0,
            "duplicates_prevented": 0,
            "by_category": {},
            "by_method": {},
            "errors": [],
            "output_dir": output_dir
        }

        # Scan images directory
        images_dir_path = Path(images_dir)
        if not images_dir_path.exists():
            results["errors"].append(f"Images directory not found: {images_dir}")
            return results

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in images_dir_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            results["errors"].append(f"No images found in {images_dir}")
            return results

        results["total_images"] = len(image_files)
        logger.info(f"Found {len(image_files)} images in {images_dir}")

        # Create output directories for each object type
        output_dir_path = Path(output_dir)
        category_dirs = {}
        for obj_name in object_names:
            # Sanitize object name for directory
            safe_name = obj_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            cat_dir = output_dir_path / safe_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            category_dirs[obj_name] = cat_dir
            results["by_category"][obj_name] = 0

        # Counter for synthetic annotation IDs
        annotation_id_counter = 0

        # Process each image
        for img_idx, img_path in enumerate(image_files):
            logger.info(f"Processing image {img_idx + 1}/{len(image_files)}: {img_path.name}")

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                error_msg = f"Failed to load image: {img_path}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                continue

            img_h, img_w = image.shape[:2]

            # Initialize extraction registry for this image (deduplication per-image)
            registry = ExtractionRegistry(
                iou_threshold=dedup_cfg.iou_threshold,
                cross_category_dedup=dedup_cfg.cross_category_dedup,
                matching_strategy=dedup_cfg.matching_strategy
            )

            # Process each object name
            for obj_name in object_names:
                logger.debug(f"  Segmenting '{obj_name}' in {img_path.name}")

                # Run SAM3 text prompt segmentation
                instances = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=obj_name,
                    min_area=min_object_area,
                    return_all_instances=True
                )

                if instances is None or len(instances) == 0:
                    logger.debug(f"    No instances of '{obj_name}' found")
                    continue

                logger.info(f"    Found {len(instances)} instance(s) of '{obj_name}'")

                # Extract each instance
                for inst_idx, instance in enumerate(instances):
                    annotation_id_counter += 1

                    # Check for duplicates
                    if dedup_cfg.enabled:
                        is_dup, dup_id, iou = registry.is_duplicate(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            category_id=0,  # synthetic
                            category_name=obj_name
                        )

                        if is_dup:
                            logger.debug(
                                f"      Instance {inst_idx} is duplicate (IoU={iou:.3f} with previous), skipping"
                            )
                            results["duplicates_prevented"] += 1
                            continue

                    # Extract object with mask
                    try:
                        extracted_img, mask_img, bbox_extracted = self._extract_object_with_mask(
                            image=image,
                            mask=instance.mask,
                            padding=padding
                        )

                        if extracted_img is None:
                            logger.warning(f"      Failed to extract instance {inst_idx}")
                            results["failed_extractions"] += 1
                            continue

                        # Save extracted image
                        safe_obj_name = obj_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        output_filename = f"{img_path.stem}_{safe_obj_name}_instance{inst_idx:03d}.png"
                        output_path = category_dirs[obj_name] / output_filename

                        cv2.imwrite(str(output_path), extracted_img)

                        # Save individual COCO JSON if requested
                        if save_individual_coco:
                            json_filename = output_path.stem + ".json"
                            json_path = category_dirs[obj_name] / json_filename

                            # Get polygon from mask
                            polygons = self.mask_to_polygon(instance.mask, simplify=True, tolerance=2.0)

                            coco_json = {
                                "info": {
                                    "description": f"Extracted {obj_name} from {img_path.name}",
                                    "date_created": datetime.now().isoformat()
                                },
                                "images": [{
                                    "id": 1,
                                    "file_name": output_filename,
                                    "width": extracted_img.shape[1],
                                    "height": extracted_img.shape[0]
                                }],
                                "annotations": [{
                                    "id": annotation_id_counter,
                                    "image_id": 1,
                                    "category_id": 1,
                                    "bbox": bbox_extracted,
                                    "area": int(np.sum(instance.mask > 0)),
                                    "segmentation": polygons if polygons else [],
                                    "iscrowd": 0
                                }],
                                "categories": [{
                                    "id": 1,
                                    "name": obj_name,
                                    "supercategory": "object"
                                }]
                            }

                            with open(json_path, 'w') as f:
                                json.dump(coco_json, f, indent=2)

                        # Register extraction
                        registry.register_extraction(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            annotation_id=annotation_id_counter,
                            category_id=0,
                            category_name=obj_name,
                            method="sam3_text_prompt"
                        )

                        # Update stats
                        results["total_objects_extracted"] += 1
                        results["by_category"][obj_name] += 1
                        results["by_method"]["sam3_text_prompt"] = results["by_method"].get("sam3_text_prompt", 0) + 1

                        logger.debug(f"      Extracted instance {inst_idx} → {output_path.name}")

                    except Exception as e:
                        error_msg = f"Error extracting instance {inst_idx} of '{obj_name}' from {img_path.name}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["failed_extractions"] += 1

            # Progress callback
            if progress_callback:
                progress_callback({
                    "extracted": results["total_objects_extracted"],
                    "failed": results["failed_extractions"],
                    "duplicates_prevented": results["duplicates_prevented"],
                    "total_images": results["total_images"],
                    "current_image": img_path.name,
                    "image_progress": f"{img_idx + 1}/{len(image_files)}"
                })

            # Memory cleanup after processing each image
            del image
            if (img_idx + 1) % 5 == 0:  # Every 5 images
                gc.collect()
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final cleanup
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save extraction summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        summary = {
            "extraction_date": start_time.isoformat(),
            "images_dir": images_dir,
            "output_dir": output_dir,
            "object_names": object_names,
            "total_images_processed": results["total_images"],
            "total_objects_extracted": results["total_objects_extracted"],
            "failed_extractions": results["failed_extractions"],
            "duplicates_prevented": results["duplicates_prevented"],
            "by_category": results["by_category"],
            "by_method": results["by_method"],
            "errors": results["errors"],
            "processing_time_seconds": processing_time,
            "settings": {
                "padding": padding,
                "min_object_area": min_object_area,
                "deduplication_enabled": dedup_cfg.enabled,
                "deduplication_iou_threshold": dedup_cfg.iou_threshold
            }
        }

        summary_path = output_dir_path / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Extraction complete: {results['total_objects_extracted']} objects extracted")
        logger.info(f"Summary saved to: {summary_path}")

        results["success"] = True
        results["processing_time_seconds"] = processing_time
        results["summary_path"] = str(summary_path)

        return results

    async def extract_custom_objects(
        self,
        images_dir: str,
        output_dir: str,
        object_names: List[str],
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication_config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Extract custom objects using text prompts only (no COCO JSON).
        Async wrapper that runs in thread pool.

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for extracted objects
            object_names: List of object names to segment
            padding: Pixels of padding around objects
            min_object_area: Minimum object area in pixels
            save_individual_coco: Save individual COCO JSON per object
            deduplication_config: Deduplication configuration dict
            progress_callback: Progress callback function

        Returns:
            Extraction results dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_custom_objects_sync,
                images_dir=images_dir,
                output_dir=output_dir,
                object_names=object_names,
                padding=padding,
                min_object_area=min_object_area,
                save_individual_coco=save_individual_coco,
                deduplication_config=deduplication_config,
                progress_callback=progress_callback
            )
        )
