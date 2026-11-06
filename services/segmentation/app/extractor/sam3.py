"""
SAM3 segmentation from bounding boxes and from text prompts.
"""

import logging
import asyncio
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Sam3Mixin:
    """SAM3 segmentation from bounding boxes and from text prompts."""

    def _segment_with_sam3_sync(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Synchronous SAM3 segmentation (runs in thread pool).

        Args:
            image: Image as numpy array (BGR or RGB)
            bbox: Bounding box [x, y, width, height]

        Returns:
            Binary mask as numpy array, or None if failed
        """
        if not self.sam3_available:
            logger.warning("SAM3 not available for segmentation")
            return None

        # Validate image
        if image is None or image.size == 0:
            logger.error("segment_with_sam3: received empty or None image")
            return None

        import torch

        try:
            h, w = image.shape[:2]
            x, y, bw, bh = [int(v) for v in bbox]

            # Validate bbox dimensions
            if bw <= 0 or bh <= 0:
                logger.warning(f"segment_with_sam3: invalid bbox dimensions: {bbox}")
                return None

            # Compute intersection of bbox with image bounds and clamp
            x1_valid = max(0, x)
            y1_valid = max(0, y)
            x2_valid = min(w, x + bw)
            y2_valid = min(h, y + bh)

            # Check if there's a valid intersection
            if x1_valid >= x2_valid or y1_valid >= y2_valid:
                logger.info(f"segment_with_sam3: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping.")
                return None

            # Use clamped coordinates for SAM3
            x, y = x1_valid, y1_valid
            bw, bh = x2_valid - x1_valid, y2_valid - y1_valid

            # Convert to RGB PIL image
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Unexpected image shape: {image.shape}")
                return None

            pil_image = Image.fromarray(image_rgb)

            # SAM3 uses box format [x1, y1, x2, y2] with labels (1=positive, 0=negative)
            input_box = [x, y, x + bw, y + bh]
            input_boxes = [[input_box]]  # [batch, num_boxes, 4]
            input_boxes_labels = [[1]]    # 1 = positive box

            # Process with SAM3
            inputs = self.sam3_processor(
                images=pil_image,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam3_model(**inputs)

            # Post-process using SAM3's instance segmentation method
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )

            if len(results) > 0 and len(results[0].get("masks", [])) > 0:
                # Get best mask (highest score)
                masks = results[0]["masks"]
                scores = results[0].get("scores", [])

                if len(scores) > 0:
                    best_idx = scores.index(max(scores))
                    mask = masks[best_idx]
                else:
                    mask = masks[0]

                # Convert to numpy - mask is already a tensor of shape (H, W)
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                else:
                    mask_np = np.array(mask).astype(np.uint8) * 255

                # Ensure correct size
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                return mask_np

            logger.warning("SAM3 returned no masks")
            return None

        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}")
            return None

    async def segment_with_sam3(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Use SAM3 to segment an object within a bounding box.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            image: Image as numpy array (BGR or RGB)
            bbox: Bounding box [x, y, width, height]

        Returns:
            Binary mask as numpy array, or None if failed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._segment_with_sam3_sync,
            image,
            bbox
        )

    def _segment_with_sam3_text_prompt_sync(
        self,
        image: np.ndarray,
        class_name: str,
        min_area: int = 100,
        return_all_instances: bool = False
    ) -> Optional[np.ndarray]:
        """
        Segment object using SAM3 with text prompt (class name).
        Synchronous implementation for thread pool execution.

        Uses SAM3's native Promptable Concept Segmentation (PCS) to detect
        and segment ALL instances of an object class using only a text description.

        Args:
            image: Input image (HxWx3, BGR format from OpenCV)
            class_name: Name of object class (e.g., "fish", "bottle", "person")
            min_area: Minimum mask area to accept
            return_all_instances: If True, return List[SAM3Instance] instead of single mask
                                 (NEW - for deduplication and multi-instance matching)

        Returns:
            If return_all_instances=False (default):
                Binary mask (HxW, uint8, 0-255) of the largest detected instance,
                or None if segmentation fails
            If return_all_instances=True:
                List[SAM3Instance] with all detected instances,
                or None if segmentation fails
        """
        if not self.sam3_available:
            logger.warning("SAM3 not available for text prompt segmentation")
            return None

        try:
            import torch
            from PIL import Image as PILImage

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = PILImage.fromarray(image_rgb)

            # Preprocess class name for better SAM3 recognition
            # Convert technical names like "Plastic_Debris" to natural language "plastic debris"
            processed_class_name = class_name.replace("_", " ").lower().strip()

            # Try with original name first, then with first word only as fallback
            text_prompts_to_try = [processed_class_name]

            # Add simpler fallback (first word only) if name has multiple words
            words = processed_class_name.split()
            if len(words) > 1:
                text_prompts_to_try.append(words[0])  # e.g., "plastic debris" → "plastic"

            logger.debug(f"SAM3 text prompt: '{class_name}' → trying prompts: {text_prompts_to_try}")

            # Try each text prompt until one succeeds
            results = None
            successful_prompt = None

            for prompt in text_prompts_to_try:
                # Prepare inputs with text prompt
                inputs = self.sam3_processor(
                    images=pil_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self.sam3_model(**inputs)

                # Post-process results to get instance masks
                temp_results = self.sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.4,  # Lower confidence threshold for more detections
                    mask_threshold=0.4,  # Lower mask threshold for more permissive segmentation
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]  # Get first (and only) image results

                # Check if any instances were found
                if "masks" in temp_results and len(temp_results["masks"]) > 0:
                    results = temp_results
                    successful_prompt = prompt
                    logger.debug(f"SAM3 text prompt: '{prompt}' found {len(results['masks'])} instance(s)")
                    break
                else:
                    logger.debug(f"SAM3 text prompt: '{prompt}' found no instances, trying next...")

            # If no prompt succeeded
            if results is None or "masks" not in results or len(results["masks"]) == 0:
                logger.warning(
                    f"SAM3 text prompt: No instances found with any prompt {text_prompts_to_try}. "
                    f"Original class: '{class_name}'. "
                    f"Suggestions: (1) Use bbox-guided SAM3 instead, (2) Check if object is visible in image, "
                    f"(3) Try more generic class names in your dataset."
                )
                return None

            # Get masks and scores
            masks = results["masks"]  # List of binary masks (H, W)
            scores = results.get("scores", [])  # List of confidence scores

            logger.info(
                f"SAM3 text prompt: Found {len(masks)} instance(s) using prompt '{successful_prompt}' "
                f"(original class: '{class_name}') with scores {scores}"
            )

            # NEW: Return all instances mode (for deduplication)
            if return_all_instances:
                from app.utils import SAM3Instance

                instances = []
                for idx, mask in enumerate(masks):
                    # Convert mask to numpy if it's a tensor
                    if hasattr(mask, 'cpu'):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)

                    # Ensure binary (0 or 255)
                    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                    # Calculate area
                    area = np.sum(mask_np > 0)

                    # Skip if too small
                    if area < min_area:
                        logger.debug(f"Mask {idx} too small: {area} < {min_area}")
                        continue

                    # Get bbox from mask
                    bbox = self.get_mask_bbox(mask_np)
                    if bbox is None:
                        logger.debug(f"Mask {idx} has no valid bbox")
                        continue

                    # Get score
                    score = scores[idx] if idx < len(scores) else 0.5

                    # Create SAM3Instance
                    instances.append(SAM3Instance(
                        mask=mask_np,
                        bbox=bbox,
                        score=score,
                        assigned=False
                    ))

                if not instances:
                    logger.warning(
                        f"SAM3 text prompt: All {len(masks)} mask(s) found with prompt '{successful_prompt}' "
                        f"(original class: '{class_name}') are below minimum area {min_area} pixels."
                    )
                    return None

                logger.info(
                    f"✓ SAM3 text prompt: Returning {len(instances)} instances for '{class_name}' "
                    f"(prompt: '{successful_prompt}')"
                )
                return instances

            # EXISTING: Return best single mask (backward compatible)
            best_mask = None
            best_area = 0
            best_score = 0

            for idx, mask in enumerate(masks):
                # Convert mask to numpy if it's a tensor
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                # Ensure binary (0 or 255)
                mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                # Calculate area
                area = np.sum(mask_np > 0)

                # Skip if too small
                if area < min_area:
                    logger.debug(f"Mask {idx} too small: {area} < {min_area}")
                    continue

                # Select mask with best combination of area and score
                score = scores[idx] if idx < len(scores) else 0.5

                # Prioritize by score, then area
                if score > best_score or (score == best_score and area > best_area):
                    best_mask = mask_np
                    best_area = area
                    best_score = score

            if best_mask is None:
                logger.warning(
                    f"SAM3 text prompt: All {len(masks)} mask(s) found with prompt '{successful_prompt}' "
                    f"(original class: '{class_name}') are below minimum area {min_area} pixels. "
                    f"Largest mask area: {max([np.sum(m > 0) if hasattr(m, '__iter__') else 0 for m in masks], default=0)}px. "
                    f"Consider: (1) Reducing min_area (currently {min_area}), (2) Using bbox-guided SAM3, "
                    f"(3) Checking if object is too small/distant in image."
                )
                return None

            logger.info(
                f"✓ SAM3 text prompt: Successfully segmented using prompt '{successful_prompt}' "
                f"(original class: '{class_name}', area={best_area}px, score={best_score:.3f})"
            )
            return best_mask

        except Exception as e:
            logger.error(f"SAM3 text prompt segmentation failed for '{class_name}': {e}", exc_info=True)
            return None

    async def segment_with_sam3_text_prompt(
        self,
        image: np.ndarray,
        class_name: str,
        min_area: int = 100
    ) -> Optional[np.ndarray]:
        """
        Segment object using SAM3 with text prompt (class name).
        Async wrapper that runs in thread pool to avoid blocking.

        Args:
            image: Input image (HxWx3, BGR or RGB)
            class_name: Name of object class (e.g., "fish", "bottle")
            min_area: Minimum mask area in pixels to accept

        Returns:
            Binary mask as numpy array (HxW, uint8, 0-255), or None if failed

        Example:
            >>> mask = await extractor.segment_with_sam3_text_prompt(
            ...     image=img,
            ...     class_name="fish",
            ...     min_area=100
            ... )
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._segment_with_sam3_text_prompt_sync,
            image,
            class_name,
            min_area
        )
