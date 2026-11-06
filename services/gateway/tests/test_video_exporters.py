"""Tests for video export formats (MOT, YouTube-VIS, CVAT video)."""

import json
import zipfile

import pytest

from app.export.native.cvat_video import export_cvat_video
from app.export.native.youtube_vis import export_youtube_vis


@pytest.fixture()
def video_coco(tmp_path):
    """COCO-extended tracking dataset: 4 frames, 2 tracks."""
    coco = {
        "videos": [{"id": 1, "name": "vid", "fps": 10.0, "frame_stride": 1,
                    "frame_count": 4, "width": 100, "height": 80}],
        "images": [
            {"id": i + 1, "file_name": f"/data/frames/{i:06d}.jpg", "width": 100,
             "height": 80, "video_id": 1, "frame_index": i}
            for i in range(4)
        ],
        "annotations": [
            # track 0: frames 0-3
            *[{"id": i + 1, "image_id": i + 1, "category_id": 1,
               "bbox": [10 + i * 5, 10, 20, 20], "area": 400, "iscrowd": 0,
               "segmentation": [[10 + i * 5, 10, 30 + i * 5, 10, 30 + i * 5, 30, 10 + i * 5, 30]],
               "score": 0.9, "track_id": 0, "video_id": 1} for i in range(4)],
            # track 1: frames 1-2 only
            *[{"id": 10 + i, "image_id": i + 1, "category_id": 2,
               "bbox": [50, 40, 15, 15], "area": 225, "iscrowd": 0,
               "segmentation": [], "score": 0.7, "track_id": 1, "video_id": 1}
              for i in (1, 2)],
        ],
        "categories": [{"id": 1, "name": "fish"}, {"id": 2, "name": "rock"}],
    }
    path = tmp_path / "video.json"
    path.write_text(json.dumps(coco))
    return {"coco": coco, "path": path}


def test_youtube_vis_tracks(video_coco, tmp_path):
    out = tmp_path / "out"
    export_youtube_vis(video_coco["coco"], out)
    data = json.loads((out / "youtube_vis.json").read_text())

    assert data["videos"][0]["length"] == 4
    assert len(data["annotations"]) == 2
    track0 = next(a for a in data["annotations"] if a["id"] == 0)
    assert len(track0["bboxes"]) == 4
    assert all(b is not None for b in track0["bboxes"])
    track1 = next(a for a in data["annotations"] if a["id"] == 1)
    assert track1["bboxes"][0] is None and track1["bboxes"][1] is not None


def test_cvat_video_tracks(video_coco, tmp_path):
    out = tmp_path / "out"
    export_cvat_video(video_coco["coco"], out)
    xml = (out / "annotations.xml").read_text()

    assert '<track id="0" label="fish">' in xml
    assert '<track id="1" label="rock">' in xml
    assert "<mode>interpolation</mode>" in xml
    assert 'outside="1"' in xml  # track 1 closes before the video ends
    assert "<polygon" in xml


def test_mot_via_engine(video_coco, tmp_path):
    datumaro = pytest.importorskip("datumaro")
    from app.export.engine import run_export

    zip_path = run_export(str(video_coco["path"]), "mot", tmp_path / "exp")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]
        gt = zf.read(next(n for n in names if n.endswith("gt.txt"))).decode()

    lines = [l for l in gt.strip().splitlines() if l]
    assert len(lines) == 6  # 4 + 2 track detections
    first = lines[0].split(",")
    assert len(first) >= 7  # frame, id, x, y, w, h, conf


def test_video_formats_registered():
    from app.export.formats import FORMATS

    for fmt in ("mot", "youtube_vis", "cvat_video"):
        assert fmt in FORMATS
        assert "video" in FORMATS[fmt].tasks
