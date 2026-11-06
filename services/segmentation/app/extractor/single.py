"""
Extraction of a single annotated object.
"""

import logging
import asyncio
import base64
from typing import Dict, Any
from functools import partial

import cv2
import numpy as np

from app.models.extraction_schemas import AnnotationType, ExtractionMethod

logger = logging.getLogger(__name__)


class SingleObjectMixin:
    """Extraction of a single annotated object."""

    def _extract_single_object_sync(
        self,
        image: np.ndarray,
        annotation: Dict[str, Any],
        category_name: str,
        use_sam3: bool = False,
        padding: int = 5,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronous version of extract_single_object (runs in thread pool).

        Args:
            image: Source image as numpy array
            annotation: COCO annotation for the object
            category_name: Name of the category
            use_sam3: Whether to use SAM3 for bbox-only annotations
            padding: Padding around extracted object
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks

        Returns:
            Dictionary with extraction results
        """
        # Validate image
        if image is None:
            return {"success": False, "error": "Image is None"}
        if not hasattr(image, 'shape') or len(image.shape) < 2:
            return {"success": False, "error": f"Invalid image shape: {getattr(image, 'shape', 'no shape')}"}
        if image.size == 0:
            return {"success": False, "error": "Image is empty (size=0)"}

        h, w = image.shape[:2]
        bbox = annotation.get("bbox", [0, 0, w, h])

        # Force SAM3 text prompt mode - ignore everything, use only class label
        if force_sam3_text_prompt and use_sam3 and self.sam3_available:
            logger.debug(f"Force text prompt mode: using only class label '{category_name}' with SAM3")

            # Use SAM3 with class name as text prompt on full image
            # Lower min_area (50px) to be more permissive with small objects
            mask = self._segment_with_sam3_text_prompt_sync(
                image=image,
                class_name=category_name,
                min_area=50
            )

            if mask is not None:
                # Calculate bbox from generated mask
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x = int(x_indices.min())
                    y = int(y_indices.min())
                    bw = int(x_indices.max() - x + 1)
                    bh = int(y_indices.max() - y + 1)
                    bbox = [x, y, bw, bh]

                    # Crop with mask
                    cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                    if cropped is None:
                        return {"success": False, "error": f"Failed to crop with text-prompted mask"}

                    # Calculate mask coverage
                    bbox_area = bw * bh
                    mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                    mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                    # Encode image
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        return {"success": False, "error": "Failed to encode image"}
                    cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                    # Encode mask
                    x1, y1, mw, mh = new_bbox
                    cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                    success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                    mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                    return {
                        "success": True,
                        "cropped_image_base64": cropped_base64,
                        "mask_base64": mask_base64,
                        "annotation_type": AnnotationType.BBOX_ONLY.value,
                        "method_used": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                        "original_bbox": bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]],
                        "mask_coverage": mask_coverage
                    }

            # If SAM3 text prompt failed
            logger.warning(f"SAM3 text prompt failed for '{category_name}' - cannot extract")
            return {"success": False, "error": "Text-prompted segmentation failed"}

        # Validate bbox
        if not bbox or len(bbox) < 4:
            return {"success": False, "error": f"Invalid bbox: {bbox}"}

        # Ensure bbox values are valid
        bx, by, bw, bh = bbox[:4]
        if bw <= 0 or bh <= 0:
            return {"success": False, "error": f"Invalid bbox dimensions: {bbox}"}

        # Check if annotation lacks both bbox and segmentation (classification-only)
        # In this case, try SAM3 with text prompt to segment the object
        has_real_bbox = annotation.get("bbox") is not None and annotation.get("bbox") != [0, 0, w, h]
        has_segmentation = annotation.get("segmentation") is not None and len(annotation.get("segmentation", [])) > 0

        if not has_real_bbox and not has_segmentation:
            # Annotation only has category_id - try text-prompted segmentation
            if use_sam3 and self.sam3_available:
                logger.debug(f"No bbox/segmentation found - trying SAM3 text prompt for '{category_name}'")

                # Attempt segmentation using class name as text prompt
                # Lower min_area (50px) to be more permissive with small objects
                mask = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=category_name,
                    min_area=50
                )

                if mask is not None:
                    # Calculate bbox from generated mask
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x = int(x_indices.min())
                        y = int(y_indices.min())
                        bw = int(x_indices.max() - x + 1)
                        bh = int(y_indices.max() - y + 1)
                        bbox = [x, y, bw, bh]

                        # Crop with mask
                        cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                        if cropped is None:
                            return {"success": False, "error": f"Failed to crop with text-prompted mask"}

                        # Calculate mask coverage
                        bbox_area = bw * bh
                        mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                        mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                        # Encode image
                        success, encoded = cv2.imencode('.png', cropped)
                        if not success:
                            return {"success": False, "error": "Failed to encode image"}
                        cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                        # Encode mask
                        x1, y1, mw, mh = new_bbox
                        cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                        success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                        mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                        return {
                            "success": True,
                            "cropped_image_base64": cropped_base64,
                            "mask_base64": mask_base64,
                            "annotation_type": AnnotationType.BBOX_ONLY.value,
                            "method_used": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                            "original_bbox": bbox,
                            "extracted_size": [cropped.shape[1], cropped.shape[0]],
                            "mask_coverage": mask_coverage
                        }

                # If SAM3 text prompt failed or returned no mask
                logger.warning(f"SAM3 text prompt segmentation failed for '{category_name}' - skipping object")
                return {"success": False, "error": "Text-prompted segmentation not available or failed"}
            else:
                # SAM3 not available and no bbox/segmentation
                logger.warning(f"Cannot extract '{category_name}' without bbox/mask and SAM3 unavailable")
                return {"success": False, "error": "No bbox/segmentation and SAM3 not available"}

        # Force bbox-only extraction if requested (ignore masks)
        if force_bbox_only:
            logger.debug(f"Force bbox-only mode: skipping mask extraction for {category_name}")
            cropped, new_bbox = self.crop_bbox_only(image, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image for bbox={bbox}"}

            # Encode to base64
            success, encoded = cv2.imencode('.png', cropped)
            if not success:
                return {"success": False, "error": "Failed to encode image"}

            cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

            return {
                "success": True,
                "cropped_image_base64": cropped_base64,
                "mask_base64": None,
                "annotation_type": AnnotationType.BBOX_ONLY.value,
                "method_used": ExtractionMethod.BBOX_CROP.value,
                "original_bbox": bbox,
                "extracted_size": [cropped.shape[1], cropped.shape[0]],
                "mask_coverage": 1.0
            }

        # Force SAM3 resegmentation if requested (regenerate masks even if they exist)
        if force_sam3_resegmentation and use_sam3 and self.sam3_available:
            # Check if annotation has existing segmentation (polygon or RLE)
            has_segmentation = annotation.get("segmentation") is not None and len(annotation.get("segmentation", [])) > 0

            if has_segmentation:
                logger.debug(f"Force SAM3 resegmentation: ignoring existing mask for {category_name}")

                # Use SAM3 with bbox to regenerate mask
                mask = self._segment_with_sam3_sync(image, bbox)

                if mask is not None:
                    # Crop with new SAM3-generated mask
                    cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                    if cropped is None:
                        return {"success": False, "error": f"Failed to crop with SAM3-regenerated mask"}

                    # Calculate mask coverage
                    x, y, bw, bh = [int(v) for v in bbox]
                    bbox_area = bw * bh
                    mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                    mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                    # Encode image
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        return {"success": False, "error": "Failed to encode image"}
                    cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                    # Encode mask
                    x1, y1, mw, mh = new_bbox
                    cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                    success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                    mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                    return {
                        "success": True,
                        "cropped_image_base64": cropped_base64,
                        "mask_base64": mask_base64,
                        "annotation_type": AnnotationType.BBOX_ONLY.value,
                        "method_used": ExtractionMethod.SAM3_FROM_BBOX.value,
                        "original_bbox": bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]],
                        "mask_coverage": mask_coverage
                    }
                else:
                    logger.warning(f"SAM3 resegmentation failed for {category_name} - falling back to original mask")
                    # Continue with normal flow to use original mask

        ann_type = self.detect_annotation_type(annotation)

        mask = None
        method = ExtractionMethod.BBOX_CROP

        # Get or generate mask
        if ann_type == AnnotationType.POLYGON:
            mask = self.polygon_to_mask(annotation["segmentation"], h, w)
            method = ExtractionMethod.POLYGON_MASK

        elif ann_type == AnnotationType.RLE:
            mask = self.rle_to_mask(annotation["segmentation"], h, w)
            method = ExtractionMethod.RLE_MASK

        elif use_sam3 and self.sam3_available:
            # Call sync version directly since we're already in a thread
            mask = self._segment_with_sam3_sync(image, bbox)
            if mask is not None:
                method = ExtractionMethod.SAM3_FROM_BBOX

        # Crop image
        if mask is not None:
            # IMPORTANT: When we have a mask from polygon/RLE, use the actual mask bbox
            # instead of the annotation bbox, to ensure correct alignment
            # This fixes issues where annotation bbox doesn't match the segmentation
            if method in [ExtractionMethod.POLYGON_MASK, ExtractionMethod.RLE_MASK]:
                mask_bbox = self.get_mask_bbox(mask)
                if mask_bbox is not None:
                    # Use mask's actual bounding box for cropping
                    bbox = mask_bbox
                    logger.debug(f"Using mask-derived bbox {mask_bbox} instead of annotation bbox {annotation.get('bbox')}")

            cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image with mask for bbox={bbox}"}

            # Calculate mask coverage
            x, y, bw, bh = [int(v) for v in bbox]
            bbox_area = bw * bh
            mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
            mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0
        else:
            cropped, new_bbox = self.crop_bbox_only(image, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image for bbox={bbox}"}

            mask_coverage = 1.0

        # Encode to base64
        success, encoded = cv2.imencode('.png', cropped)
        if not success:
            return {"success": False, "error": "Failed to encode image"}

        cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

        # Encode mask if available
        mask_base64 = None
        if mask is not None:
            x1, y1, mw, mh = new_bbox
            cropped_mask = mask[y1:y1+mh, x1:x1+mw]
            success, encoded_mask = cv2.imencode('.png', cropped_mask)
            if success:
                mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8')

        return {
            "success": True,
            "cropped_image_base64": cropped_base64,
            "mask_base64": mask_base64,
            "annotation_type": ann_type.value,
            "method_used": method.value,
            "original_bbox": bbox,
            "extracted_size": [cropped.shape[1], cropped.shape[0]],
            "mask_coverage": mask_coverage
        }

    async def extract_single_object(
        self,
        image: np.ndarray,
        annotation: Dict[str, Any],
        category_name: str,
        use_sam3: bool = False,
        padding: int = 5,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Extract a single object from an image.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            image: Source image as numpy array
            annotation: COCO annotation for the object
            category_name: Name of the category
            use_sam3: Whether to use SAM3 for bbox-only annotations
            padding: Padding around extracted object
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks

        Returns:
            Dictionary with extraction results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_single_object_sync,
                image=image,
                annotation=annotation,
                category_name=category_name,
                use_sam3=use_sam3,
                padding=padding,
                force_bbox_only=force_bbox_only,
                force_sam3_resegmentation=force_sam3_resegmentation,
                force_sam3_text_prompt=force_sam3_text_prompt
            )
        )
