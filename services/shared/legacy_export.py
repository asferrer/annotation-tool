"""
Legacy bbox-only exporters shared by gateway and segmentation.

Limitations: YOLO detection and Pascal VOC only, bboxes only (no masks).
Superseded by the Datumaro-based export engine in the gateway; kept for the
labeling pipeline's inline output_formats.
"""

from pathlib import Path
from typing import Any, Dict, List


def _annotations_by_image(coco_data: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return anns_by_image


def export_yolo(coco_data: Dict[str, Any], output_dir: Path) -> str:
    """Export COCO bboxes to YOLO detection format under output_dir/yolo."""
    labels_dir = output_dir / "yolo" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco_data.get("images", [])}
    image_names = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
    categories = coco_data.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    for img_id, anns in _annotations_by_image(coco_data).items():
        if img_id not in image_dims:
            continue
        w, h = image_dims[img_id]
        lines = []
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            lines.append(
                f"{cat_map.get(ann['category_id'], 0)} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{bbox[2] / w:.6f} {bbox[3] / h:.6f}"
            )
        label_file = labels_dir / (Path(image_names[img_id]).stem + ".txt")
        label_file.write_text("\n".join(lines))

    classes_file = output_dir / "yolo" / "classes.txt"
    classes_file.write_text("\n".join(c["name"] for c in categories))
    return str(output_dir / "yolo")


def export_voc(coco_data: Dict[str, Any], output_dir: Path) -> str:
    """Export COCO bboxes to Pascal VOC XML under output_dir/voc."""
    annotations_dir = output_dir / "voc" / "Annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_info = {img["id"]: img for img in coco_data.get("images", [])}
    cat_names = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    for img_id, anns in _annotations_by_image(coco_data).items():
        if img_id not in image_info:
            continue
        img = image_info[img_id]
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<annotation>",
            f'  <filename>{img["file_name"]}</filename>',
            "  <size>",
            f'    <width>{img["width"]}</width>',
            f'    <height>{img["height"]}</height>',
            "    <depth>3</depth>",
            "  </size>",
        ]
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            xml_lines += [
                "  <object>",
                f'    <name>{cat_names.get(ann["category_id"], "unknown")}</name>',
                "    <bndbox>",
                f"      <xmin>{int(bbox[0])}</xmin>",
                f"      <ymin>{int(bbox[1])}</ymin>",
                f"      <xmax>{int(bbox[0] + bbox[2])}</xmax>",
                f"      <ymax>{int(bbox[1] + bbox[3])}</ymax>",
                "    </bndbox>",
                "  </object>",
            ]
        xml_lines.append("</annotation>")
        xml_file = annotations_dir / (Path(img["file_name"]).stem + ".xml")
        xml_file.write_text("\n".join(xml_lines))

    return str(output_dir / "voc")
