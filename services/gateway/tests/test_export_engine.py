"""Integration tests for the export/import engine (requires datumaro)."""

import json
import zipfile

import pytest

datumaro = pytest.importorskip("datumaro")

from app.export.engine import run_export
from app.export.formats import FORMATS, get_format, list_import_formats


def _zip_names(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def test_registry_is_consistent():
    for spec in FORMATS.values():
        if spec.backend == "datumaro":
            assert spec.datumaro_name, spec.id
    assert get_format("coco").supports_import
    assert {f.id for f in list_import_formats()} >= {"coco", "voc", "labelme"}


def test_export_coco_roundtrip(coco_dataset, tmp_path):
    zip_path = run_export(str(coco_dataset["coco_path"]), "coco", tmp_path / "exp")
    names = _zip_names(zip_path)
    coco_name = next(n for n in names if n.endswith(".json"))
    with zipfile.ZipFile(zip_path) as zf:
        data = json.loads(zf.read(coco_name))
    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    assert {c["name"] for c in data["categories"]} == {"fish", "rock"}


@pytest.mark.parametrize("format_id, marker", [
    ("voc", "Annotations/img1.xml"),
    ("cvat_images", "default.xml"),
    ("labelme", "img1.xml"),
    ("dota", "img1.txt"),
    ("masks_png", "SegmentationClass/img1.png"),
])
def test_export_datumaro_formats(coco_dataset, tmp_path, format_id, marker):
    zip_path = run_export(str(coco_dataset["coco_path"]), format_id, tmp_path / "exp")
    names = [n.replace("\\", "/") for n in _zip_names(zip_path)]
    assert any(marker in n for n in names), f"{marker} not in {names}"


def test_export_native_via_engine(coco_dataset, tmp_path):
    zip_path = run_export(str(coco_dataset["coco_path"]), "yolo_seg", tmp_path / "exp")
    names = [n.replace("\\", "/") for n in _zip_names(zip_path)]
    assert "labels/train/img1.txt" in names


def test_export_with_images(coco_dataset, tmp_path):
    zip_path = run_export(
        str(coco_dataset["coco_path"]), "yolo_det", tmp_path / "exp",
        include_images=True, images_dir=str(coco_dataset["images_dir"]),
    )
    names = [n.replace("\\", "/") for n in _zip_names(zip_path)]
    assert "images/train/img1.jpg" in names


def test_export_unknown_format_raises(coco_dataset, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        run_export(str(coco_dataset["coco_path"]), "nope", tmp_path / "exp")


def test_import_voc_to_coco(coco_dataset, tmp_path, monkeypatch):
    # Export to VOC, then import it back as a new COCO task
    from app.export import importer

    zip_path = run_export(
        str(coco_dataset["coco_path"]), "voc", tmp_path / "exp",
        include_images=True, images_dir=str(coco_dataset["images_dir"]),
    )
    voc_dir = tmp_path / "voc_src"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(voc_dir)

    monkeypatch.setattr(importer, "DATASETS_BASE", tmp_path / "datasets")
    summary = importer.run_import(str(voc_dir), "voc", "reimported task")

    coco = json.loads(open(summary["coco_json_path"]).read())
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
