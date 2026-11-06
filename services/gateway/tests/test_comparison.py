"""Tests for the COCO comparison (evaluation) router."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import comparison, filesystem


def _coco(images, annotations, categories):
    return {"images": images, "annotations": annotations, "categories": categories}


def _ann(ann_id, image_id, cat, bbox, **extra):
    return {"id": ann_id, "image_id": image_id, "category_id": cat, "bbox": bbox, **extra}


class TestBboxIou:
    def test_identical_boxes(self):
        assert comparison.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert comparison.bbox_iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0

    def test_partial_overlap(self):
        # intersection 5x10 = 50, union 100 + 100 - 50 = 150
        assert comparison.bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)

    def test_degenerate_boxes(self):
        assert comparison.bbox_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


class TestMatchAnnotations:
    def test_perfect_match(self):
        gt = [_ann(1, 1, 1, [0, 0, 10, 10])]
        pred = [_ann(1, 1, 1, [0, 0, 10, 10], score=0.9)]
        assert comparison.match_annotations(gt, pred, 0.5) == (1, 0, 0)

    def test_below_threshold_is_fp_and_fn(self):
        gt = [_ann(1, 1, 1, [0, 0, 10, 10])]
        pred = [_ann(1, 1, 1, [8, 0, 10, 10], score=0.9)]
        assert comparison.match_annotations(gt, pred, 0.5) == (0, 1, 1)

    def test_ground_truth_matched_once(self):
        gt = [_ann(1, 1, 1, [0, 0, 10, 10])]
        pred = [
            _ann(1, 1, 1, [0, 0, 10, 10], score=0.9),
            _ann(2, 1, 1, [0, 0, 10, 10], score=0.8),
        ]
        assert comparison.match_annotations(gt, pred, 0.5) == (1, 1, 0)

    def test_higher_score_claims_match_first(self):
        gt = [_ann(1, 1, 1, [0, 0, 10, 10])]
        good = _ann(1, 1, 1, [0, 0, 10, 10], score=0.9)
        loose = _ann(2, 1, 1, [1, 0, 10, 10], score=0.5)
        assert comparison.match_annotations(gt, [loose, good], 0.5) == (1, 1, 0)


class TestPrf1:
    def test_values(self):
        m = comparison.prf1(tp=3, fp=1, fn=2)
        assert m["precision"] == 0.75
        assert m["recall"] == 0.6
        assert m["f1"] == pytest.approx(0.6667, abs=1e-4)

    def test_empty_counts(self):
        m = comparison.prf1(0, 0, 0)
        assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)


class TestEvaluateCoco:
    def test_matches_images_and_categories_by_name(self):
        gt = _coco(
            [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            [_ann(1, 1, 1, [0, 0, 10, 10]), _ann(2, 2, 2, [0, 0, 10, 10])],
            [{"id": 1, "name": "Fish"}, {"id": 2, "name": "rock"}],
        )
        # Different image ids and category ids; names differ only by case.
        pred = _coco(
            [{"id": 7, "file_name": "a.jpg"}, {"id": 8, "file_name": "b.jpg"}],
            [
                _ann(1, 7, 5, [0, 0, 10, 10], score=0.9),   # fish: TP
                _ann(2, 8, 6, [50, 50, 5, 5], score=0.9),   # rock: FP (and its GT is FN)
                _ann(3, 7, 9, [0, 0, 5, 5], score=0.9),     # unknown category: dropped
            ],
            [
                {"id": 5, "name": "fish"},
                {"id": 6, "name": "ROCK"},
                {"id": 9, "name": "unknown"},
            ],
        )
        result = comparison.evaluate_coco(gt, pred, 0.5)

        assert result["matched_images"] == 2
        assert result["global_metrics"]["tp"] == 1
        assert result["global_metrics"]["fp"] == 1
        assert result["global_metrics"]["fn"] == 1
        by_name = {c["category_name"]: c for c in result["per_category"]}
        assert by_name["Fish"]["f1"] == 1.0
        assert by_name["rock"]["tp"] == 0

    def test_image_missing_from_predictions_counts_as_fn(self):
        gt = _coco(
            [{"id": 1, "file_name": "a.jpg"}],
            [_ann(1, 1, 1, [0, 0, 10, 10])],
            [{"id": 1, "name": "fish"}],
        )
        pred = _coco([], [], [{"id": 1, "name": "fish"}])
        result = comparison.evaluate_coco(gt, pred, 0.5)

        assert result["matched_images"] == 0
        assert result["global_metrics"]["fn"] == 1
        assert result["global_metrics"]["recall"] == 0.0

    def test_annotations_without_bbox_are_ignored(self):
        gt = _coco(
            [{"id": 1, "file_name": "a.jpg"}],
            [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": []}],
            [{"id": 1, "name": "fish"}],
        )
        result = comparison.evaluate_coco(gt, gt, 0.5)
        assert result["per_category"] == []
        assert result["global_metrics"]["tp"] == 0


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(filesystem, "ALLOWED_BASE_PATHS", [tmp_path.resolve()])
    app = FastAPI()
    app.include_router(comparison.router)
    return TestClient(app)


class TestEvaluateEndpoint:
    def _write(self, path: Path, data) -> str:
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_evaluate(self, client, tmp_path):
        data = _coco(
            [{"id": 1, "file_name": "a.jpg"}],
            [_ann(1, 1, 1, [0, 0, 10, 10])],
            [{"id": 1, "name": "fish"}],
        )
        gt = self._write(tmp_path / "gt.json", data)
        pred = self._write(tmp_path / "pred.json", data)

        resp = client.post(
            "/comparison/evaluate",
            json={"ground_truth_path": gt, "predictions_path": pred, "iou_threshold": 0.5},
        )
        assert resp.status_code == 200
        assert resp.json()["global_metrics"]["f1"] == 1.0

    def test_missing_file(self, client, tmp_path):
        resp = client.post(
            "/comparison/evaluate",
            json={
                "ground_truth_path": str(tmp_path / "nope.json"),
                "predictions_path": str(tmp_path / "nope2.json"),
            },
        )
        assert resp.status_code == 404

    def test_path_outside_allowed_directories(self, client):
        resp = client.post(
            "/comparison/evaluate",
            json={"ground_truth_path": "/outside/allowed/gt.json", "predictions_path": "/outside/allowed/gt.json"},
        )
        assert resp.status_code == 403

    def test_invalid_json(self, client, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        resp = client.post(
            "/comparison/evaluate",
            json={"ground_truth_path": str(bad), "predictions_path": str(bad)},
        )
        assert resp.status_code == 400

    def test_invalid_threshold(self, client, tmp_path):
        resp = client.post(
            "/comparison/evaluate",
            json={
                "ground_truth_path": str(tmp_path / "a.json"),
                "predictions_path": str(tmp_path / "b.json"),
                "iou_threshold": 5,
            },
        )
        assert resp.status_code == 422
