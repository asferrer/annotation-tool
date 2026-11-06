"""
Native CVAT for video 1.1 exporter: <track> elements with per-frame boxes
and polygons (interpolation mode). Datumaro's cvat plugin only emits the
per-image variant, so tracks are written here.
"""

from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape, quoteattr


def _polygon_attr(segmentation: List[List[float]]) -> str:
    poly = segmentation[0]
    points = ";".join(
        f"{poly[i]:.2f},{poly[i + 1]:.2f}" for i in range(0, len(poly) - 1, 2)
    )
    return points


def _track_xml(track_id: int, label: str, anns: List[Dict[str, Any]], frame_of: Dict[int, int], last_frame: int) -> List[str]:
    lines = [f'  <track id="{track_id}" label={quoteattr(label)}>']
    ordered = sorted(anns, key=lambda a: frame_of[a["image_id"]])
    for i, ann in enumerate(ordered):
        frame = frame_of[ann["image_id"]]
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        # outside=1 marks the frame after the last appearance (CVAT convention)
        next_frame = frame_of[ordered[i + 1]["image_id"]] if i + 1 < len(ordered) else None
        lines.append(
            f'    <box frame="{frame}" xtl="{x:.2f}" ytl="{y:.2f}" '
            f'xbr="{x + w:.2f}" ybr="{y + h:.2f}" outside="0" occluded="0" keyframe="1">'
            "</box>"
        )
        segmentation = ann.get("segmentation") or []
        if segmentation and isinstance(segmentation[0], list) and len(segmentation[0]) >= 6:
            lines.append(
                f'    <polygon frame="{frame}" points="{_polygon_attr(segmentation)}" '
                'outside="0" occluded="0" keyframe="1"></polygon>'
            )
        if next_frame is None and frame < last_frame:
            lines.append(
                f'    <box frame="{frame + 1}" xtl="{x:.2f}" ytl="{y:.2f}" '
                f'xbr="{x + w:.2f}" ybr="{y + h:.2f}" outside="1" occluded="0" keyframe="1">'
                "</box>"
            )
    lines.append("  </track>")
    return lines


def export_cvat_video(coco: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = sorted(
        coco.get("images", []), key=lambda img: img.get("frame_index", img["id"] - 1)
    )
    frame_of = {img["id"]: img.get("frame_index", pos) for pos, img in enumerate(images)}
    last_frame = max(frame_of.values(), default=0)
    video = (coco.get("videos") or [{}])[0]

    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("track_id") is None or ann.get("image_id") not in frame_of:
            continue
        tracks.setdefault(ann["track_id"], []).append(ann)

    labels = sorted({categories.get(anns[0]["category_id"], "unknown") for anns in tracks.values()})
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        "  <version>1.1</version>",
        "  <meta>",
        "    <task>",
        f"      <name>{escape(str(video.get('name', 'video')))}</name>",
        f"      <size>{len(images)}</size>",
        "      <mode>interpolation</mode>",
        "      <start_frame>0</start_frame>",
        f"      <stop_frame>{last_frame}</stop_frame>",
        "      <labels>",
        *[f"        <label><name>{escape(l)}</name></label>" for l in labels],
        "      </labels>",
        "    </task>",
        "  </meta>",
    ]
    for track_id in sorted(tracks):
        anns = tracks[track_id]
        label = categories.get(anns[0]["category_id"], "unknown")
        lines.extend(_track_xml(track_id, label, anns, frame_of, last_frame))
    lines.append("</annotations>")

    (out_dir / "annotations.xml").write_text("\n".join(lines), encoding="utf-8")
    return out_dir
