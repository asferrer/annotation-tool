"""
Annotation-type detection and mask/polygon/RLE conversion helpers.
"""

import logging
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from app.models.extraction_schemas import AnnotationType

logger = logging.getLogger(__name__)


class MaskMixin:
    """Annotation-type detection and mask/polygon/RLE conversion helpers."""

    @staticmethod
    def detect_annotation_type(annotation: Dict[str, Any]) -> AnnotationType:
        """
        Detect the type of annotation (polygon, RLE, or bbox_only).

        Args:
            annotation: COCO annotation dictionary

        Returns:
            AnnotationType enum value
        """
        segmentation = annotation.get("segmentation")

        if segmentation is None:
            return AnnotationType.BBOX_ONLY

        # Empty list or empty segmentation
        if isinstance(segmentation, list) and len(segmentation) == 0:
            return AnnotationType.BBOX_ONLY

        # Polygon format: list of lists with coordinates
        if isinstance(segmentation, list) and len(segmentation) > 0:
            first_seg = segmentation[0]
            if isinstance(first_seg, list) and len(first_seg) >= 6:
                return AnnotationType.POLYGON

        # RLE format: dictionary with 'counts' and 'size'
        if isinstance(segmentation, dict):
            if "counts" in segmentation and "size" in segmentation:
                return AnnotationType.RLE

        return AnnotationType.BBOX_ONLY

    @staticmethod
    def polygon_to_mask(
        segmentation: List[List[float]],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert COCO polygon segmentation to binary mask.

        Args:
            segmentation: List of polygons, each polygon is [x1,y1,x2,y2,...]
            height: Image height
            width: Image width

        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in segmentation:
            if len(polygon) < 6:
                continue

            # Reshape flat list to array of points
            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        return mask

    @staticmethod
    def rle_to_mask(
        rle: Dict[str, Any],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert COCO RLE segmentation to binary mask.

        Args:
            rle: RLE dictionary with 'counts' and 'size'
            height: Image height
            width: Image width

        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        try:
            from pycocotools import mask as mask_util

            # Ensure proper format
            if isinstance(rle["counts"], list):
                # Uncompressed RLE
                rle_obj = mask_util.frPyObjects([rle], height, width)[0]
            else:
                # Compressed RLE
                rle_obj = rle

            mask = mask_util.decode(rle_obj)
            return (mask * 255).astype(np.uint8)

        except ImportError:
            logger.warning("pycocotools not available, using fallback RLE decoder")
            return MaskMixin._decode_rle_fallback(rle, height, width)

    @staticmethod
    def _decode_rle_fallback(
        rle: Dict[str, Any],
        height: int,
        width: int
    ) -> np.ndarray:
        """Fallback RLE decoder when pycocotools is not available."""
        counts = rle.get("counts", [])
        size = rle.get("size", [height, width])

        if isinstance(counts, str):
            # Compressed format - simplified decoding
            logger.warning("Compressed RLE requires pycocotools for accurate decoding")
            return np.zeros((height, width), dtype=np.uint8)

        # Uncompressed RLE
        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
        pos = 0
        val = 0

        for count in counts:
            mask[pos:pos + count] = val * 255
            pos += count
            val = 1 - val

        return mask.reshape(size).T

    @staticmethod
    def mask_to_polygon(
        mask: np.ndarray,
        simplify: bool = True,
        tolerance: float = 2.0
    ) -> List[List[float]]:
        """
        Convert binary mask to COCO polygon format.

        Args:
            mask: Binary mask (0 or 255)
            simplify: Whether to simplify the polygon
            tolerance: Simplification tolerance

        Returns:
            List of polygons in COCO format
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue

            if simplify and tolerance > 0:
                epsilon = tolerance
                contour = cv2.approxPolyDP(contour, epsilon, True)

            if len(contour) < 3:
                continue

            # Flatten to COCO format [x1,y1,x2,y2,...]
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)

        return polygons

    @staticmethod
    def get_mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
        """
        Get the bounding box of non-zero pixels in a mask.

        Args:
            mask: Binary mask (0 or 255)

        Returns:
            Bounding box [x, y, width, height] or None if mask is empty
        """
        # Find non-zero pixels
        coords = np.where(mask > 0)
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            return None

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
