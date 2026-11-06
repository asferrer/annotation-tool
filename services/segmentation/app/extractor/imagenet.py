"""
Extraction from ImageNet-style folder structures.
"""

import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import partial

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagenetMixin:
    """Extraction from ImageNet-style folder structures."""

    def _extract_from_imagenet_structure_sync(
        self,
        root_dir: str,
        output_dir: str,
        padding: int = 5,
        min_object_area: int = 100,
        max_objects_per_class: Optional[int] = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract objects from ImageNet-style directory structure (synchronous).

        Expected structure:
            root_dir/
            ├── class1/
            │   ├── img001.jpg
            │   ├── img002.jpg
            │   └── ...
            ├── class2/
            │   └── ...

        For each image:
        1. Use SAM3 with class name as text prompt
        2. Extract segmented object
        3. Save to output_dir/class_name/

        Args:
            root_dir: Root directory with class subdirectories
            output_dir: Output directory for extracted objects
            padding: Padding around extracted objects
            min_object_area: Minimum area filter
            max_objects_per_class: Limit objects per class (None=all)
            progress_callback: Callback for progress updates

        Returns:
            Extraction summary with stats per class
        """
        if not self.sam3_available:
            logger.error("SAM3 is required for ImageNet structure extraction but not available")
            return {
                "success": False,
                "error": "SAM3 not available",
                "total_classes": 0,
                "classes": {},
                "total_extracted": 0,
                "total_failed": 0
            }

        logger.info(f"Starting ImageNet extraction from {root_dir}")

        # Discover classes (subdirectories)
        try:
            classes = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
        except Exception as e:
            logger.error(f"Failed to list root directory: {e}")
            return {
                "success": False,
                "error": f"Failed to access root directory: {e}",
                "total_classes": 0,
                "classes": {},
                "total_extracted": 0,
                "total_failed": 0
            }

        results = {
            "success": True,
            "total_classes": len(classes),
            "classes": {},
            "total_extracted": 0,
            "total_failed": 0,
            "errors": []
        }

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            # List images in class directory
            try:
                image_files = [f for f in os.listdir(class_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            except Exception as e:
                logger.error(f"Failed to list class directory {class_name}: {e}")
                results["errors"].append(f"Class '{class_name}': {str(e)}")
                continue

            if max_objects_per_class:
                image_files = image_files[:max_objects_per_class]

            class_stats = {
                "total_images": len(image_files),
                "extracted": 0,
                "failed": 0
            }

            logger.info(f"Processing class '{class_name}': {len(image_files)} images")

            for img_idx, img_file in enumerate(image_files):
                try:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)

                    if img is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        class_stats["failed"] += 1
                        continue

                    # Segment with SAM3 text prompt
                    mask = self._segment_with_sam3_text_prompt_sync(
                        image=img,
                        class_name=class_name,
                        min_area=min_object_area
                    )

                    if mask is None:
                        class_stats["failed"] += 1
                        continue

                    # Calculate bbox from mask
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        class_stats["failed"] += 1
                        continue

                    x = int(x_indices.min())
                    y = int(y_indices.min())
                    w = int(x_indices.max() - x + 1)
                    h = int(y_indices.max() - y + 1)
                    bbox = [x, y, w, h]

                    # Extract object
                    cropped, new_bbox = self.crop_with_mask(img, mask, bbox, padding)

                    if cropped is None:
                        class_stats["failed"] += 1
                        continue

                    # Save extracted object
                    output_filename = f"{Path(img_file).stem}.png"
                    output_path = os.path.join(output_class_dir, output_filename)
                    cv2.imwrite(output_path, cropped)

                    class_stats["extracted"] += 1
                    results["total_extracted"] += 1

                    # Progress callback
                    if progress_callback and (img_idx + 1) % 10 == 0:
                        progress_callback({
                            "current_class": class_name,
                            "class_progress": f"{img_idx + 1}/{len(image_files)}",
                            "total_progress": f"{class_idx + 1}/{len(classes)} classes",
                            "extracted": results["total_extracted"],
                            "failed": results["total_failed"]
                        })

                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")
                    class_stats["failed"] += 1
                    results["total_failed"] += 1
                    results["errors"].append(f"{class_name}/{img_file}: {str(e)}")

            results["classes"][class_name] = class_stats
            results["total_failed"] += class_stats["failed"]

            logger.info(
                f"Class '{class_name}': {class_stats['extracted']}/{class_stats['total_images']} extracted"
            )

        logger.info(
            f"ImageNet extraction complete: {results['total_extracted']} extracted, "
            f"{results['total_failed']} failed from {results['total_classes']} classes"
        )

        return results

    async def extract_from_imagenet_structure(
        self,
        root_dir: str,
        output_dir: str,
        padding: int = 5,
        min_object_area: int = 100,
        max_objects_per_class: Optional[int] = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract objects from ImageNet-style directory structure (async).

        Args:
            root_dir: Root directory with class subdirectories
            output_dir: Output directory for extracted objects
            padding: Padding around extracted objects
            min_object_area: Minimum object area filter
            max_objects_per_class: Limit objects per class (None=all)
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with extraction results and statistics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_from_imagenet_structure_sync,
                root_dir=root_dir,
                output_dir=output_dir,
                padding=padding,
                min_object_area=min_object_area,
                max_objects_per_class=max_objects_per_class,
                progress_callback=progress_callback
            )
        )
