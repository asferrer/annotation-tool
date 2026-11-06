"""
Native YOLO exporters: detection, segmentation and OBB (Ultralytics layout).

All variants write labels/<stem>.txt with normalized coordinates, plus
data.yaml and classes.txt. The whole dataset goes into a single "train"
split — exporting annotations must never re-split the user's data.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _prepare(coco: Dict[str, Any], out_dir: Path) -> Tuple[Path, Dict, Dict, Dict]:
    labels_dir = out_dir / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in coco.get("images", [])}
    categories = coco.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    _write_metadata(out_dir, categories)
    return labels_dir, images, cat_map, anns_by_image


def _write_metadata(out_dir: Path, categories: List[Dict]) -> None:
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(categories))
    (out_dir / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/train\nnames:\n{names}\n"
    )
    (out_dir / "classes.txt").write_text("\n".join(c["name"] for c in categories))


def _write_labels(
    labels_dir: Path, images: Dict, anns_by_image: Dict, line_fn
) -> None:
    for img_id, anns in anns_by_image.items():
        if img_id not in images:
            continue
        img = images[img_id]
        w, h = img["width"], img["height"]
        lines = [line for ann in anns for line in line_fn(ann, w, h)]
        stem = Path(img["file_name"]).stem
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))


def export_yolo_det(coco: Dict[str, Any], out_dir: Path) -> Path:
    """YOLO detection: class cx cy w h (normalized)."""
    labels_dir, images, cat_map, anns_by_image = _prepare(coco, out_dir)

    def det_lines(ann: Dict, w: int, h: int) -> List[str]:
        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        return [
            f"{cat_map.get(ann['category_id'], 0)} "
            f"{(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
        ]

    _write_labels(labels_dir, images, anns_by_image, det_lines)
    return out_dir


def export_yolo_seg(coco: Dict[str, Any], out_dir: Path) -> Path:
    """YOLO segmentation: class x1 y1 x2 y2 ... (normalized polygon)."""
    labels_dir, images, cat_map, anns_by_image = _prepare(coco, out_dir)

    def seg_lines(ann: Dict, w: int, h: int) -> List[str]:
        segmentation = ann.get("segmentation") or []
        polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
        if not polygons:
            return []
        cls = cat_map.get(ann["category_id"], 0)
        lines = []
        for poly in polygons:
            coords = " ".join(
                f"{poly[i] / w:.6f} {poly[i + 1] / h:.6f}"
                for i in range(0, len(poly) - 1, 2)
            )
            lines.append(f"{cls} {coords}")
        return lines

    _write_labels(labels_dir, images, anns_by_image, seg_lines)
    return out_dir


def _obb_corners(ann: Dict) -> List[float]:
    """Minimum rotated rectangle from the polygon, else axis-aligned bbox corners."""
    segmentation = ann.get("segmentation") or []
    polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
    if polygons:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon

            points = [(p[i], p[i + 1]) for p in polygons for i in range(0, len(p) - 1, 2)]
            rect = ShapelyPolygon(points).minimum_rotated_rectangle
            corners = list(rect.exterior.coords)[:4]
            return [coord for point in corners for coord in point]
        except Exception:
            pass
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def export_yolo_obb(coco: Dict[str, Any], out_dir: Path) -> Path:
    """YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized corners)."""
    labels_dir, images, cat_map, anns_by_image = _prepare(coco, out_dir)

    def obb_lines(ann: Dict, w: int, h: int) -> List[str]:
        corners = _obb_corners(ann)
        coords = " ".join(
            f"{corners[i] / w:.6f} {corners[i + 1] / h:.6f}"
            for i in range(0, 8, 2)
        )
        return [f"{cat_map.get(ann['category_id'], 0)} {coords}"]

    _write_labels(labels_dir, images, anns_by_image, obb_lines)
    return out_dir
