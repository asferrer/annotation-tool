"""Shared fixtures for gateway tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make `app` and `shared` importable when running from services/gateway
GATEWAY_ROOT = Path(__file__).resolve().parents[1]
SERVICES_ROOT = GATEWAY_ROOT.parent
for path in (str(GATEWAY_ROOT), str(SERVICES_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture()
def coco_dataset(tmp_path: Path) -> dict:
    """Minimal COCO dataset on disk: 2 images, bbox + polygon annotations."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("img1.jpg", "img2.jpg"):
        Image.fromarray(
            np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        ).save(images_dir / name)

    coco = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 120, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 120, "height": 100},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 10, 30, 40], "area": 1200, "iscrowd": 0,
                "segmentation": [[10, 10, 40, 10, 40, 50, 10, 50]],
            },
            {
                "id": 2, "image_id": 2, "category_id": 2,
                "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "fish"}, {"id": 2, "name": "rock"}],
    }
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps(coco))
    return {"coco": coco, "coco_path": coco_path, "images_dir": images_dir, "root": tmp_path}
