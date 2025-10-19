import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image


def pil_from_bytes(img_bytes: bytes) -> Image.Image:
    """Decode bytes into RGB PIL image."""
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def xyxy_to_xywh(
    b: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    """Convert (x1,y1,x2,y2) to (x,y,w,h)."""
    x1, y1, x2, y2 = map(float, b)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def load_class_names_from_yaml(yaml_path: str) -> List[str]:
    """Load YOLO-style class names from dataset YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not isinstance(names, list) or not names:
        raise ValueError("Missing 'names' in dataset YAML.")
    return [str(n) for n in names]


def left_right_neighbors_for_gap(
    gap_bbox: Dict[str, int],
    products: List[Dict[str, Any]],
    min_vertical_overlap: float = 0.3,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return nearest left/right products for the given gap."""
    gx1, gy1 = gap_bbox["x"], gap_bbox["y"]
    gw, gh = gap_bbox["w"], gap_bbox["h"]
    gx2, gy2 = gx1 + gw, gy1 + gh

    left_best = None
    right_best = None

    def vo_frac(py1, py2) -> float:
        overlap = max(0, min(gy2, py2) - max(gy1, py1))
        return overlap / max(1, min(gh, (py2 - py1)))

    # First pass: same shelf (vertical overlap)
    for p in products:
        b = p["bbox"]
        px1, py1 = b["x"], b["y"]
        pw, ph = b["w"], b["h"]
        px2, py2 = px1 + pw, py1 + ph
        if vo_frac(py1, py2) < min_vertical_overlap:
            continue
        if px2 <= gx1:  # left candidate
            d = gx1 - px2
            if left_best is None or d < left_best[0]:
                left_best = (d, p)
        if px1 >= gx2:  # right candidate
            d = px1 - gx2
            if right_best is None or d < right_best[0]:
                right_best = (d, p)

    # Second pass: ignore shelf (no vertical overlap filter)
    if left_best is None or right_best is None:
        for p in products:
            b = p["bbox"]
            px1, py1 = b["x"], b["y"]
            pw = b["w"]
            px2 = px1 + pw
            if left_best is None and px2 <= gx1:
                d = gx1 - px2
                if left_best is None or d < left_best[0]:
                    left_best = (d, p)
            if right_best is None and px1 >= gx2:
                d = px1 - gx2
                if right_best is None or d < right_best[0]:
                    right_best = (d, p)

    return {
        "left": None if left_best is None else left_best[1],
        "right": None if right_best is None else right_best[1],
    }


def vote_missing_skus(
    gaps: List[Dict[str, Any]],
    all_products: List[Dict[str, Any]],
    min_vertical_overlap: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (neighbor_products_with_side, gap_neighbors) and update votes."""
    votes: Dict[str, float] = {}
    for g in gaps:
        sides = left_right_neighbors_for_gap(
            g["bbox"],
            all_products,
            min_vertical_overlap=min_vertical_overlap,
        )
        g["neighbors"] = {
            "left": sides["left"],
            "right": sides["right"],
        }

        # vote weighting by horizontal distance and detector score
        gx1 = g["bbox"]["x"]
        gx2 = g["bbox"]["x"] + g["bbox"]["w"]
        for side in ("left", "right"):
            p = sides[side]
            if p is None:
                continue
            b = p["bbox"]
            px1, px2 = b["x"], b["x"] + b["w"]
            dist = max(0, (gx1 - px2) if side == "left" else (px1 - gx2))
            weight = p.get("score", 1.0) / (1.0 + float(dist))
            votes[p["sku_id"]] = votes.get(p["sku_id"], 0.0) + weight

    return votes


def combine_votes_with_freq(
    votes: Dict[str, float], products: List[Dict[str, Any]], alpha: float = 0.6
) -> List[Dict[str, Any]]:
    """Combine normalized neighbor votes with image-level frequency."""
    freq: Dict[str, int] = {}
    for p in products:
        freq[p["sku_id"]] = freq.get(p["sku_id"], 0) + 1
    max_votes = max(votes.values()) if votes else 1.0
    max_freq = max(freq.values()) if freq else 1
    combined = []
    for k in set(list(votes.keys()) + list(freq.keys())):
        v = (votes.get(k, 0.0) / max_votes) * alpha + (
            freq.get(k, 0) / max_freq
        ) * (1 - alpha)
        combined.append({"sku_id": k, "confidence": round(float(v), 2)})
    combined.sort(key=lambda d: d["confidence"], reverse=True)
    return combined[:2]


def occupancy_from_gaps(
    img_w: int, img_h: int, gaps: List[Dict[str, Any]]
) -> float:
    """Compute occupancy = 1 - (gap_area / image_area)."""
    total = max(1, img_w * img_h)
    gap_area = sum(g["area_px"] for g in gaps)
    return max(0.0, min(1.0, round(1.0 - (gap_area / total), 2)))
