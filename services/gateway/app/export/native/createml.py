"""
Native CreateML exporter (Apple object detection JSON).

Output: createml.json — a list of {image, annotations: [{label, coordinates}]}
where coordinates use the box CENTER in absolute pixels.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def export_createml(coco: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in coco.get("images", [])}
    cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}

    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    entries = []
    for img_id, img in images.items():
        annotations = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            annotations.append({
                "label": cat_names.get(ann["category_id"], "unknown"),
                "coordinates": {
                    "x": round(x + w / 2, 2),
                    "y": round(y + h / 2, 2),
                    "width": round(w, 2),
                    "height": round(h, 2),
                },
            })
        entries.append({"image": Path(img["file_name"]).name, "annotations": annotations})

    (out_dir / "createml.json").write_text(json.dumps(entries, indent=2))
    return out_dir
