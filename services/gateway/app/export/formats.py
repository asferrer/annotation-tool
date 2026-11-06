"""
Format registry: single source of truth for supported export/import formats.

backend "datumaro": handled by datumaro Dataset.export with optional transforms.
backend "native": handled by exporters in app/export/native/ (formats datumaro
cannot produce faithfully, e.g. YOLO segmentation/OBB, CreateML, YouTube-VIS).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class FormatSpec(BaseModel):
    id: str
    label: str
    backend: Literal["datumaro", "native"]
    datumaro_name: Optional[str] = None
    transforms: List[str] = []
    tasks: List[str] = ["bbox"]
    supports_import: bool = False
    import_name: Optional[str] = None
    description: str = ""


FORMATS: Dict[str, FormatSpec] = {
    spec.id: spec
    for spec in [
        FormatSpec(
            id="coco",
            label="COCO JSON",
            backend="datumaro",
            datumaro_name="coco_instances",
            tasks=["bbox", "polygon"],
            supports_import=True,
            import_name="coco_instances",
            description="COCO instances JSON (bbox + polygon segmentation)",
        ),
        FormatSpec(
            id="yolo_det",
            label="YOLO Detection",
            backend="native",
            tasks=["bbox"],
            supports_import=True,
            import_name="yolo",
            description="Ultralytics layout: labels/*.txt (class cx cy w h) + data.yaml",
        ),
        FormatSpec(
            id="yolo_seg",
            label="YOLO Segmentation",
            backend="native",
            tasks=["polygon"],
            description="Ultralytics segmentation: labels/*.txt with normalized polygons",
        ),
        FormatSpec(
            id="yolo_obb",
            label="YOLO OBB",
            backend="native",
            tasks=["bbox", "polygon"],
            description="Oriented bounding boxes: 4 normalized corners per line",
        ),
        FormatSpec(
            id="voc",
            label="Pascal VOC XML",
            backend="datumaro",
            datumaro_name="voc_detection",
            tasks=["bbox"],
            supports_import=True,
            import_name="voc",
            description="One XML per image with bndbox objects",
        ),
        FormatSpec(
            id="masks_png",
            label="Segmentation masks (PNG)",
            backend="datumaro",
            datumaro_name="voc_segmentation",
            transforms=["polygons_to_masks"],
            tasks=["polygon"],
            description="Semantic (SegmentationClass) + instance (SegmentationObject) PNG masks",
        ),
        FormatSpec(
            id="cvat_images",
            label="CVAT XML (images)",
            backend="datumaro",
            datumaro_name="cvat",
            tasks=["bbox", "polygon"],
            supports_import=True,
            import_name="cvat",
            description="CVAT for images 1.1 XML",
        ),
        FormatSpec(
            id="labelme",
            label="LabelMe",
            backend="datumaro",
            datumaro_name="label_me",
            tasks=["bbox", "polygon"],
            supports_import=True,
            import_name="label_me",
            description="LabelMe XML per image with polygon shapes",
        ),
        FormatSpec(
            id="dota",
            label="DOTA",
            backend="datumaro",
            datumaro_name="dota",
            tasks=["bbox", "polygon"],
            supports_import=True,
            import_name="dota",
            description="Aerial imagery format: labelTxt/*.txt with 4-corner polygons",
        ),
        FormatSpec(
            id="createml",
            label="CreateML JSON",
            backend="native",
            tasks=["bbox"],
            supports_import=True,
            import_name="roboflow_create_ml",
            description="Apple CreateML object detection JSON (center-based coordinates)",
        ),
        FormatSpec(
            id="mot",
            label="MOT Challenge",
            backend="datumaro",
            datumaro_name="mot_seq_gt",
            tasks=["video"],
            description="Multi-object tracking: gt/gt.txt (frame, track id, bbox, conf)",
        ),
        FormatSpec(
            id="youtube_vis",
            label="YouTube-VIS",
            backend="native",
            tasks=["video"],
            description="Video instance segmentation: one entry per track with per-frame masks",
        ),
        FormatSpec(
            id="cvat_video",
            label="CVAT XML (video)",
            backend="native",
            tasks=["video"],
            description="CVAT for video 1.1: <track> elements with per-frame boxes/polygons",
        ),
    ]
}


def get_format(format_id: str) -> FormatSpec:
    if format_id not in FORMATS:
        raise ValueError(
            f"Unknown format '{format_id}'. Supported: {sorted(FORMATS)}"
        )
    return FORMATS[format_id]


def list_formats() -> List[FormatSpec]:
    return list(FORMATS.values())


def list_import_formats() -> List[FormatSpec]:
    return [f for f in FORMATS.values() if f.supports_import]
