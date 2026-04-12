"""
Comparison Router - Compare two COCO JSON annotations (ground truth vs predictions).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Comparison"], prefix="/comparison")


class CompareRequest(BaseModel):
    ground_truth_path: str
    predictions_path: str
    iou_threshold: float = Field(default=0.5, ge=0.1, le=1.0)


def _load_coco(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_area(bbox: List[float]) -> float:
    return bbox[2] * bbox[3]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _match_annotations(
    gt_anns: List[dict],
    pred_anns: List[dict],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0

    sorted_preds = sorted(
        pred_anns,
        key=lambda a: a.get("score", a.get("area", 0)),
        reverse=True,
    )

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_anns):
            if i in matched_gt:
                continue
            if gt.get("category_id") != pred.get("category_id"):
                continue
            iou = _bbox_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_anns) - tp
    fn = len(gt_anns) - len(matched_gt)
    return tp, fp, fn


def _build_category_map(categories: List[dict]) -> Dict[int, str]:
    return {c["id"]: c["name"] for c in categories}


def _map_categories_by_name(
    gt_cats: List[dict], pred_cats: List[dict]
) -> Dict[int, int]:
    gt_name_to_id = {c["name"].lower(): c["id"] for c in gt_cats}
    return {
        c["id"]: gt_name_to_id[c["name"].lower()]
        for c in pred_cats
        if c["name"].lower() in gt_name_to_id
    }


def _remap_annotations(
    anns: List[dict], cat_map: Dict[int, int]
) -> List[dict]:
    remapped = []
    for a in anns:
        new_cat = cat_map.get(a.get("category_id"))
        if new_cat is not None:
            remapped.append({**a, "category_id": new_cat})
    return remapped


@router.post("/evaluate")
async def evaluate(request: CompareRequest):
    gt_data = _load_coco(request.ground_truth_path)
    pred_data = _load_coco(request.predictions_path)

    gt_cats = gt_data.get("categories", [])
    pred_cats = pred_data.get("categories", [])
    cat_name_map = _build_category_map(gt_cats)

    pred_to_gt_cat = _map_categories_by_name(gt_cats, pred_cats)
    pred_anns = _remap_annotations(
        pred_data.get("annotations", []), pred_to_gt_cat
    )
    gt_anns = gt_data.get("annotations", [])

    gt_by_image: Dict[int, List[dict]] = defaultdict(list)
    for a in gt_anns:
        if "bbox" in a:
            gt_by_image[a["image_id"]].append(a)

    pred_by_image: Dict[int, List[dict]] = defaultdict(list)
    gt_img_filenames = {
        img["id"]: img["file_name"] for img in gt_data.get("images", [])
    }
    pred_img_filenames = {
        img["file_name"]: img["id"] for img in pred_data.get("images", [])
    }
    gt_id_to_pred_id = {}
    for gt_img in gt_data.get("images", []):
        pred_id = pred_img_filenames.get(gt_img["file_name"])
        if pred_id is not None:
            gt_id_to_pred_id[gt_img["id"]] = pred_id

    for a in pred_anns:
        if "bbox" in a:
            pred_by_image[a["image_id"]].append(a)

    all_image_ids = set(gt_by_image.keys())
    total_tp, total_fp, total_fn = 0, 0, 0
    per_cat_tp: Dict[int, int] = defaultdict(int)
    per_cat_fp: Dict[int, int] = defaultdict(int)
    per_cat_fn: Dict[int, int] = defaultdict(int)

    for gt_img_id in all_image_ids:
        pred_img_id = gt_id_to_pred_id.get(gt_img_id)
        img_gt = gt_by_image.get(gt_img_id, [])
        img_pred = pred_by_image.get(pred_img_id, []) if pred_img_id else []

        cat_ids_in_image = set(
            a["category_id"] for a in img_gt + img_pred
        )
        for cat_id in cat_ids_in_image:
            cat_gt = [a for a in img_gt if a["category_id"] == cat_id]
            cat_pred = [a for a in img_pred if a["category_id"] == cat_id]
            tp, fp, fn = _match_annotations(
                cat_gt, cat_pred, request.iou_threshold
            )
            per_cat_tp[cat_id] += tp
            per_cat_fp[cat_id] += fp
            per_cat_fn[cat_id] += fn
            total_tp += tp
            total_fp += fp
            total_fn += fn

    def _metrics(tp: int, fp: int, fn: int) -> dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    per_category = []
    for cat_id in sorted(per_cat_tp.keys() | per_cat_fp.keys() | per_cat_fn.keys()):
        m = _metrics(per_cat_tp[cat_id], per_cat_fp[cat_id], per_cat_fn[cat_id])
        m["category_id"] = cat_id
        m["category_name"] = cat_name_map.get(cat_id, f"id_{cat_id}")
        per_category.append(m)

    matched_images = len(gt_id_to_pred_id)

    return {
        "iou_threshold": request.iou_threshold,
        "matched_images": matched_images,
        "total_gt_images": len(gt_data.get("images", [])),
        "total_pred_images": len(pred_data.get("images", [])),
        "global_metrics": _metrics(total_tp, total_fp, total_fn),
        "per_category": per_category,
        "gt_categories": [{"id": c["id"], "name": c["name"]} for c in gt_cats],
        "pred_categories": [{"id": c["id"], "name": c["name"]} for c in pred_cats],
    }
