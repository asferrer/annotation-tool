"""
Import engine: any registered format -> normalized COCO dataset in DATASETS_BASE.

Uses datumaro importers and re-exports to coco_instances, producing
<datasets>/<task_name>/annotations/instances_default.json (+ images/ when the
source dataset bundles media).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from .formats import get_format

logger = logging.getLogger(__name__)

DATASETS_BASE = Path(os.environ.get("DATASETS_BASE", "/app/datasets"))


def _sanitize_task_name(task_name: str) -> str:
    safe = "".join(c for c in task_name if c.isalnum() or c in ("-", "_", " ")).strip()
    if not safe:
        raise ValueError(f"Invalid task name: {task_name!r}")
    return safe.replace(" ", "_")


def run_import(input_path: str, format_id: str, task_name: str) -> Dict[str, Any]:
    """Convert a dataset to COCO under DATASETS_BASE/task_name. Returns summary."""
    import datumaro as dm

    spec = get_format(format_id)
    if not spec.supports_import or not spec.import_name:
        raise ValueError(f"Format '{format_id}' does not support import")

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    dataset = dm.Dataset.import_from(str(source), spec.import_name)
    out_root = DATASETS_BASE / _sanitize_task_name(task_name)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        dataset.export(str(out_root), "coco_instances", save_media=True)
        media_saved = True
    except Exception as e:
        logger.warning(f"Import without media (source media unavailable): {e}")
        dataset.export(str(out_root), "coco_instances", save_media=False)
        media_saved = False

    coco_files = sorted((out_root / "annotations").glob("instances_*.json"))
    if not coco_files:
        raise RuntimeError("Import produced no COCO annotations file")

    return {
        "coco_json_path": str(coco_files[0]),
        "items": len(dataset),
        "media_saved": media_saved,
        "output_dir": str(out_root),
    }
