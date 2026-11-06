"""Unit tests for native exporters (YOLO det/seg/OBB, CreateML)."""

import json

from app.export.native.createml import export_createml
from app.export.native.yolo import export_yolo_det, export_yolo_obb, export_yolo_seg


def test_yolo_det_labels_normalized(coco_dataset, tmp_path):
    out = tmp_path / "out"
    export_yolo_det(coco_dataset["coco"], out)

    label = (out / "labels" / "train" / "img1.txt").read_text().split()
    assert label[0] == "0"  # category fish -> index 0
    values = [float(v) for v in label[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
    # cx = (10 + 15) / 120
    assert abs(values[0] - 25 / 120) < 1e-4
    assert (out / "data.yaml").exists()
    assert (out / "classes.txt").read_text().splitlines() == ["fish", "rock"]


def test_yolo_seg_writes_polygons_and_skips_bbox_only(coco_dataset, tmp_path):
    out = tmp_path / "out"
    export_yolo_seg(coco_dataset["coco"], out)

    seg_line = (out / "labels" / "train" / "img1.txt").read_text().split()
    assert seg_line[0] == "0"
    assert len(seg_line) == 1 + 8  # 4 polygon points x 2 coords
    # img2 has no polygon -> empty label file
    assert (out / "labels" / "train" / "img2.txt").read_text() == ""


def test_yolo_obb_four_corners(coco_dataset, tmp_path):
    out = tmp_path / "out"
    export_yolo_obb(coco_dataset["coco"], out)

    for name in ("img1.txt", "img2.txt"):
        line = (out / "labels" / "train" / name).read_text().split()
        assert len(line) == 1 + 8
        assert all(0.0 <= float(v) <= 1.0 for v in line[1:])


def test_createml_center_coordinates(coco_dataset, tmp_path):
    out = tmp_path / "out"
    export_createml(coco_dataset["coco"], out)

    entries = json.loads((out / "createml.json").read_text())
    assert len(entries) == 2
    first = next(e for e in entries if e["image"] == "img1.jpg")
    coords = first["annotations"][0]["coordinates"]
    assert coords == {"x": 25.0, "y": 30.0, "width": 30.0, "height": 40.0}
