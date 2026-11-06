"""
Smoke tests for the segmentation API surface (no GPU, no external services).

The route table is compared against a stored snapshot so that refactors of the
routers cannot silently add, drop or rename an endpoint.
"""

import json
import os

from fastapi.testclient import TestClient

from app.main import app

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routes_snapshot.json")


def _route_table():
    rows = set()
    for route in app.routes:
        for method in getattr(route, "methods", None) or []:
            if method != "HEAD":
                rows.add((method, route.path))
    return sorted(rows)


def test_route_table_matches_snapshot():
    with open(SNAPSHOT, encoding="utf-8") as fh:
        expected = sorted(tuple(row) for row in json.load(fh))
    assert _route_table() == expected


def test_ping_and_health_without_lifespan():
    client = TestClient(app)  # not used as a context manager: lifespan (GPU/SAM3) is skipped
    assert client.get("/ping").json() == {"status": "ok"}
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert "sam3_available" in health


def test_model_status_reports_components():
    body = TestClient(app).get("/model-status").json()
    assert set(body) == {"sam3", "gpu", "scene_analyzer", "object_extractor"}


def test_openapi_document_builds():
    schema = app.openapi()
    assert "/labeling/start" in schema["paths"]
    assert "/health" in schema["paths"]


def test_unknown_labeling_job_returns_404():
    response = TestClient(app).get("/labeling/jobs/does-not-exist")
    assert response.status_code == 404


def test_object_extractor_public_api():
    from app.object_extractor import ObjectExtractor

    expected = {
        "analyze_dataset", "convert_bbox_to_segmentation", "create_individual_coco",
        "crop_bbox_only", "crop_with_mask", "detect_annotation_type",
        "extract_custom_objects", "extract_from_dataset",
        "extract_from_imagenet_structure", "extract_single_object", "get_mask_bbox",
        "mask_to_polygon", "polygon_to_mask", "resolve_image_path", "rle_to_mask",
        "segment_with_sam3", "segment_with_sam3_text_prompt",
    }
    assert expected <= set(dir(ObjectExtractor))
    extractor = ObjectExtractor()
    assert extractor.sam3_available is False
