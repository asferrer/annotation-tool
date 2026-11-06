"""
Comparison Router - Evaluate predicted COCO annotations against a ground truth.

Matching is greedy per image and per category: predictions are visited in
descending confidence and each one claims the unmatched ground-truth box with
the highest IoU, provided it reaches the threshold. Images are paired by
``file_name`` and categories by (case-insensitive) name, so the two files do
not need to share ids.
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.routers.filesystem import _validate_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Comparison"], prefix="/comparison")


class CompareRequest(BaseModel):
    ground_truth_path: str
    predictions_path: str
    iou_threshold: float = Field(default=0.5, ge=0.1, le=1.0)


# ---------- Helpers ----------

def _load_coco(path: str) -> Dict[str, Any]:
    p = _validate_path(path)
    if not p.is_file():
        raise HTTPException(404, f"File not found: {path}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in {path}: {e}")


def bbox_iou(a: List[float], b: List[float]) -> float:
    """IoU of two COCO ``[x, y, w, h]`` boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_annotations(
    gt_anns: List[dict], pred_anns: List[dict], iou_threshold: float
) -> Tuple[int, int, int]:
    """Greedy matching of one image/category. Returns (tp, fp, fn)."""
    matched_gt = set()
    ranked = sorted(
        pred_anns, key=lambda a: a.get("score", a.get("area", 0)), reverse=True
    )

    for pred in ranked:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_anns):
            if i in matched_gt:
                continue
            iou = bbox_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)

    tp = len(matched_gt)
    return tp, len(pred_anns) - tp, len(gt_anns) - tp


def prf1(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    """Precision / recall / F1 from raw counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _remap_prediction_categories(
    gt_cats: List[dict], pred_cats: List[dict], pred_anns: List[dict]
) -> List[dict]:
    """Rewrite prediction category ids to ground-truth ids by name.

    Predictions whose category does not exist in the ground truth are dropped.
    """
    gt_ids = {c["name"].lower(): c["id"] for c in gt_cats}
    pred_to_gt = {
        c["id"]: gt_ids[c["name"].lower()]
        for c in pred_cats
        if c["name"].lower() in gt_ids
    }
    return [
        {**a, "category_id": pred_to_gt[a["category_id"]]}
        for a in pred_anns
        if a.get("category_id") in pred_to_gt
    ]


def _group_by_image(anns: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for a in anns:
        if "bbox" in a:
            grouped[a["image_id"]].append(a)
    return grouped


# ---------- Evaluation ----------

def evaluate_coco(gt: Dict[str, Any], pred: Dict[str, Any], iou_threshold: float) -> Dict[str, Any]:
    """Compare two COCO dicts and return global and per-category metrics."""
    gt_cats = gt.get("categories", [])
    pred_cats = pred.get("categories", [])
    cat_names = {c["id"]: c["name"] for c in gt_cats}

    pred_anns = _remap_prediction_categories(gt_cats, pred_cats, pred.get("annotations", []))
    gt_by_image = _group_by_image(gt.get("annotations", []))
    pred_by_image = _group_by_image(pred_anns)

    pred_ids_by_name = {img["file_name"]: img["id"] for img in pred.get("images", [])}
    gt_to_pred_image = {
        img["id"]: pred_ids_by_name[img["file_name"]]
        for img in gt.get("images", [])
        if img["file_name"] in pred_ids_by_name
    }

    counts: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0])
    for gt_img_id, img_gt in gt_by_image.items():
        pred_img_id = gt_to_pred_image.get(gt_img_id)
        img_pred = pred_by_image.get(pred_img_id, []) if pred_img_id is not None else []
        for cat_id in {a["category_id"] for a in img_gt + img_pred}:
            tp, fp, fn = match_annotations(
                [a for a in img_gt if a["category_id"] == cat_id],
                [a for a in img_pred if a["category_id"] == cat_id],
                iou_threshold,
            )
            c = counts[cat_id]
            c[0] += tp
            c[1] += fp
            c[2] += fn

    per_category = [
        {
            **prf1(*counts[cat_id]),
            "category_id": cat_id,
            "category_name": cat_names.get(cat_id, f"id_{cat_id}"),
        }
        for cat_id in sorted(counts)
    ]
    totals = [sum(c[i] for c in counts.values()) for i in range(3)]

    return {
        "iou_threshold": iou_threshold,
        "matched_images": len(gt_to_pred_image),
        "total_gt_images": len(gt.get("images", [])),
        "total_pred_images": len(pred.get("images", [])),
        "global_metrics": prf1(*totals),
        "per_category": per_category,
        "gt_categories": [{"id": c["id"], "name": c["name"]} for c in gt_cats],
        "pred_categories": [{"id": c["id"], "name": c["name"]} for c in pred_cats],
    }


@router.post("/evaluate")
async def evaluate(request: CompareRequest) -> Dict[str, Any]:
    """Evaluate a predictions COCO file against a ground-truth COCO file."""
    gt = await asyncio.to_thread(_load_coco, request.ground_truth_path)
    pred = await asyncio.to_thread(_load_coco, request.predictions_path)
    return await asyncio.to_thread(evaluate_coco, gt, pred, request.iou_threshold)
