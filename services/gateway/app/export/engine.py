"""
Export engine: COCO JSON -> any registered format, zipped.

Datumaro-backed formats are staged as a coco_instances dataset in a temp dir,
imported, transformed and exported. Native formats are written directly from
the parsed COCO dict. Both paths end in a single zip artifact.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formats import FormatSpec, get_format
from .native import NATIVE_EXPORTERS

logger = logging.getLogger(__name__)


def _load_coco(coco_json_path: str) -> Dict[str, Any]:
    path = Path(coco_json_path)
    if not path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_json_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image(file_name: str, roots: List[Path]) -> Optional[Path]:
    """Find an image by file_name under the candidate roots (fuzzy by basename)."""
    for root in roots:
        direct = root / file_name
        if direct.exists():
            return direct
    name = Path(file_name).name
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob(name):
            return candidate
    return None


def _copy_images(coco: Dict[str, Any], roots: List[Path], dest: Path) -> int:
    """Copy dataset images into dest, preserving basenames. Returns copied count."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in coco.get("images", []):
        src = _resolve_image(img["file_name"], roots)
        if src is None:
            logger.warning(f"Image not found for export: {img['file_name']}")
            continue
        shutil.copy2(src, dest / Path(img["file_name"]).name)
        copied += 1
    return copied


def _stage_coco_dataset(coco: Dict[str, Any], staging: Path) -> None:
    """Write a coco_instances-importable tree: annotations/ + images/default/."""
    ann_dir = staging / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    # The coco importer requires an images/ dir even with save_media=False
    (staging / "images" / "default").mkdir(parents=True, exist_ok=True)
    # Basenames only so datumaro item ids stay flat and media resolves in images/default
    flat = {**coco, "images": [
        {**img, "file_name": Path(img["file_name"]).name} for img in coco.get("images", [])
    ]}
    (ann_dir / "instances_default.json").write_text(json.dumps(flat), encoding="utf-8")


def _dataset_from_video_coco(coco: Dict[str, Any]):
    """Build a datumaro dataset preserving track_id as annotation attribute."""
    import datumaro as dm
    from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Polygon
    from datumaro.components.media import Image

    categories = coco.get("categories", [])
    label_cats = LabelCategories()
    label_index = {}
    for i, cat in enumerate(categories):
        label_cats.add(cat["name"])
        label_index[cat["id"]] = i

    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    items = []
    images = sorted(coco.get("images", []), key=lambda i: i.get("frame_index", i["id"] - 1))
    for img in images:
        annotations = []
        for ann in anns_by_image.get(img["id"], []):
            label = label_index.get(ann["category_id"], 0)
            attrs = {"track_id": int(ann.get("track_id", 0)), "keyframe": True}
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            annotations.append(Bbox(x, y, w, h, label=label, attributes=dict(attrs)))
            for poly in ann.get("segmentation") or []:
                if isinstance(poly, list) and len(poly) >= 6:
                    annotations.append(Polygon(poly, label=label, attributes=dict(attrs)))
        frame = img.get("frame_index", img["id"] - 1)
        items.append(dm.DatasetItem(
            id=Path(img["file_name"]).stem,
            media=Image.from_file(path=img["file_name"], size=(img["height"], img["width"])),
            annotations=annotations,
            attributes={"frame": frame},
        ))
    return dm.Dataset.from_iterable(items, categories={AnnotationType.label: label_cats})


def _export_datumaro(
    coco: Dict[str, Any],
    spec: FormatSpec,
    work_dir: Path,
    image_roots: List[Path],
    include_images: bool,
) -> None:
    import datumaro as dm

    if "video" in spec.tasks:
        dataset = _dataset_from_video_coco(coco)
        for transform in spec.transforms:
            dataset.transform(transform)
        dataset.export(str(work_dir), spec.datumaro_name, save_media=False)
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        _stage_coco_dataset(coco, staging)
        if include_images:
            _copy_images(coco, image_roots, staging / "images" / "default")

        dataset = dm.Dataset.import_from(str(staging), "coco_instances")
        for transform in spec.transforms:
            dataset.transform(transform)
        dataset.export(str(work_dir), spec.datumaro_name, save_media=include_images)


def _export_native(
    coco: Dict[str, Any],
    spec: FormatSpec,
    work_dir: Path,
    image_roots: List[Path],
    include_images: bool,
) -> None:
    exporter = NATIVE_EXPORTERS[spec.id]
    exporter(coco, work_dir)
    if include_images:
        subdir = "images/train" if spec.id.startswith("yolo") else "images"
        _copy_images(coco, image_roots, work_dir / subdir)


def run_export(
    coco_json_path: str,
    format_id: str,
    out_dir: Path,
    include_images: bool = False,
    images_dir: Optional[str] = None,
) -> Path:
    """Export a COCO file to the requested format. Returns the zip path."""
    spec = get_format(format_id)
    coco = _load_coco(coco_json_path)

    image_roots = [Path(coco_json_path).parent]
    if images_dir:
        image_roots.insert(0, Path(images_dir))

    work_dir = out_dir / format_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    if spec.backend == "native":
        _export_native(coco, spec, work_dir, image_roots, include_images)
    else:
        _export_datumaro(coco, spec, work_dir, image_roots, include_images)

    zip_base = out_dir / f"{format_id}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", work_dir))
    shutil.rmtree(work_dir, ignore_errors=True)
    return zip_path
