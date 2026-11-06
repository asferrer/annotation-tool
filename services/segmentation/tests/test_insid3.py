"""Tests for the INSID3 integration (CPU-only: schemas, rasterization, vendor imports)."""

import numpy as np
import pytest
from PIL import Image

from app.insid3.engine import _base_coco, _list_targets, _reference_mask
from app.insid3.loader import weights_status
from app.insid3.schemas import PropagateRequest


def _make_request(tmp_path, **overrides):
    ref = tmp_path / "ref.jpg"
    Image.fromarray(np.zeros((60, 80, 3), dtype=np.uint8)).save(ref)
    targets = tmp_path / "targets"
    targets.mkdir(exist_ok=True)
    for name in ("a.jpg", "b.png"):
        Image.fromarray(np.zeros((60, 80, 3), dtype=np.uint8)).save(targets / name)
    defaults = dict(
        reference_image_path=str(ref),
        reference_polygon=[[10, 10, 40, 10, 40, 40, 10, 40]],
        target_directory=str(targets),
        label="fish",
        output_json=str(tmp_path / "out.json"),
    )
    defaults.update(overrides)
    return PropagateRequest(**defaults)


def test_vendor_imports():
    from app.insid3.vendor import INSID3  # noqa: F401
    from app.insid3.vendor.clustering import agglomerative_clustering  # noqa: F401


def test_reference_mask_rasterizes_polygon(tmp_path):
    request = _make_request(tmp_path)
    mask = _reference_mask(request)
    arr = np.array(mask)
    assert arr.shape == (60, 80)
    assert arr[25, 25] == 255 and arr[5, 5] == 0


def test_reference_mask_out_of_bounds_polygon_raises(tmp_path):
    # Fully outside the image -> rasterizes empty -> must raise
    request = _make_request(
        tmp_path, reference_polygon=[[-20, -20, -10, -20, -10, -10, -20, -10]]
    )
    with pytest.raises(ValueError):
        _reference_mask(request)


def test_list_targets_excludes_reference(tmp_path):
    request = _make_request(tmp_path)
    targets = _list_targets(request)
    names = {t.name for t in targets}
    assert names == {"a.jpg", "b.png"}


def test_list_targets_max_images(tmp_path):
    request = _make_request(tmp_path, max_images=1)
    assert len(_list_targets(request)) == 1


def test_base_coco_single_category(tmp_path):
    request = _make_request(tmp_path)
    coco = _base_coco(request)
    assert coco["categories"] == [{"id": 1, "name": "fish"}]


def test_weights_status_reports_missing():
    status = weights_status()
    assert "weights_path" in status
    if not status["weights_present"]:
        assert "Download" in status["instructions"]
