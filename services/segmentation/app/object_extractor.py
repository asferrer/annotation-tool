"""
Object Extractor
================
Extract objects from COCO datasets as transparent PNG images.

Supports:
- Direct extraction using polygon/RLE segmentation masks
- SAM3-based segmentation for bbox-only annotations
- Individual COCO JSON generation per object

The implementation lives in the ``app.extractor`` package, split by concern into
mixins; this module keeps the public ``ObjectExtractor`` class.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from app.extractor.masks import MaskMixin
from app.extractor.sam3 import Sam3Mixin
from app.extractor.cropping import CropMixin
from app.extractor.single import SingleObjectMixin
from app.extractor.dataset import DatasetMixin
from app.extractor.dataset_modes import DatasetModesMixin
from app.extractor.imagenet import ImagenetMixin
from app.extractor.conversion import ConversionMixin
from app.extractor.custom import CustomObjectsMixin

logger = logging.getLogger(__name__)


class ObjectExtractor(
    MaskMixin,
    Sam3Mixin,
    CropMixin,
    SingleObjectMixin,
    DatasetMixin,
    DatasetModesMixin,
    ImagenetMixin,
    ConversionMixin,
    CustomObjectsMixin,
):
    """
    Extract objects from a COCO dataset as PNG images with transparency.

    Uses existing segmentation masks when available, or SAM3 for bbox-only annotations.
    """

    def __init__(
        self,
        sam3_model=None,
        sam3_processor=None,
        device: str = "cpu"
    ):
        """
        Initialize the ObjectExtractor.

        Args:
            sam3_model: Pre-loaded SAM3 model instance (from service state)
            sam3_processor: Pre-loaded SAM3 processor instance (from service state)
            device: Device to use for SAM3 inference ("cuda" or "cpu")
        """
        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor
        self.device = device
        self.sam3_available = sam3_model is not None and sam3_processor is not None

        # Thread pool for CPU-bound operations to avoid blocking the event loop
        # Using max_workers=2 to limit concurrent heavy operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"ObjectExtractor initialized (SAM3 available: {self.sam3_available})")
