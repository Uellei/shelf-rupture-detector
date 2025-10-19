# pipeline.py
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

from .utils import (
    combine_votes_with_freq,
    load_class_names_from_yaml,
    occupancy_from_gaps,
    pil_from_bytes,
    vote_missing_skus,
    xyxy_to_xywh,
)

DATASET_YAML_PATH = "configs/market-products-v2.yaml"
PRODUCT_WEIGHTS = "runs/11s_aug_v13/weights/best.pt"
GAP_WEIGHTS = "runs/empty_shelves_v1/weights/best.pt"
IMGSZ = 640
WARMUP_IMGSZ = 320
MIN_VERTICAL_OVERLAP = 0.3


def occupancy_from_gaps(
    img_w: int, img_h: int, gaps: List[Dict[str, Any]]
) -> float:
    """Compute occupancy = 1 - (gap_area / image_area), clamped to [0,1]."""
    total_area = max(1, img_w * img_h)
    gap_area = sum(g["area_px"] for g in gaps)
    return max(0.0, min(1.0, round(1.0 - (gap_area / total_area), 2)))


class InferencePipeline:
    """Load YOLO models once and run inference for gaps and neighbor products."""

    def __init__(self) -> None:
        try:
            self.class_names = load_class_names_from_yaml(DATASET_YAML_PATH)
        except Exception as e:
            print(f"[WARN] Could not load class names: {e}")
            self.class_names = []

        t0 = time.time()
        self.product_model = YOLO(PRODUCT_WEIGHTS)
        self.gap_model = YOLO(GAP_WEIGHTS)
        self.version = (
            f"product:{self.product_model.__class__.__name__}@ultralytics | "
            f"gap:{self.gap_model.__class__.__name__}@ultralytics"
        )

        self._warmup()
        self.load_ms = int((time.time() - t0) * 1000)

        prod_nc = getattr(self.product_model.model, "nc", None)
        if (
            isinstance(prod_nc, int)
            and self.class_names
            and prod_nc != len(self.class_names)
        ):
            print(
                f"[WARN] product model nc={prod_nc} != len(class_names)={len(self.class_names)}"
            )

    def _warmup(self) -> None:
        """Run a tiny forward pass to prime weights/kernels."""
        dummy = Image.fromarray(
            np.zeros((WARMUP_IMGSZ, WARMUP_IMGSZ, 3), dtype=np.uint8)
        )
        _ = self.product_model.predict(dummy, imgsz=WARMUP_IMGSZ, verbose=False)
        _ = self.gap_model.predict(dummy, imgsz=WARMUP_IMGSZ, verbose=False)

    def _sku_id(self, class_id: int) -> str:
        """Map class index to SKU id string."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"SKU_{class_id}"

    def infer(
        self,
        img_bytes: bytes,
        min_conf_gap: float = 0.5,
        min_conf_prod: float = 0.5,
        min_vertical_overlap: float = MIN_VERTICAL_OVERLAP,
    ) -> Dict[str, Any]:
        """Run detectors, attach neighbors inside each gap, compute hypotheses & occupancy."""
        img = pil_from_bytes(img_bytes)

        # Product detection
        r_prod = self.product_model.predict(
            img, conf=min_conf_prod, imgsz=IMGSZ, verbose=False
        )[0]
        all_products: List[Dict[str, Any]] = []
        if getattr(r_prod, "boxes", None) is not None and len(r_prod.boxes) > 0:
            xyxy = r_prod.boxes.xyxy.cpu().numpy()
            cls = r_prod.boxes.cls.cpu().numpy()
            conf = r_prod.boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cls, conf):
                x, y, w, h = xyxy_to_xywh(b)
                all_products.append(
                    {
                        "sku_id": self._sku_id(int(c)),
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "score": float(round(float(s), 3)),
                    }
                )

        # Gap detection
        r_gap = self.gap_model.predict(
            img, conf=min_conf_gap, imgsz=IMGSZ, verbose=False
        )[0]
        gaps: List[Dict[str, Any]] = []
        if getattr(r_gap, "boxes", None) is not None and len(r_gap.boxes) > 0:
            g_xyxy = r_gap.boxes.xyxy.cpu().numpy()
            g_conf = r_gap.boxes.conf.cpu().numpy()
            for b, s in zip(g_xyxy, g_conf):
                x, y, w, h = xyxy_to_xywh(b)
                area = int(w * h)
                ar = round(w / h if h > 0 else 0.0, 2)
                gaps.append(
                    {
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "final_score": float(round(float(s), 3)),
                        "area_px": area,
                        "aspect_ratio": ar,
                    }
                )

        # Attach neighbors inside each gap + collect votes
        votes = vote_missing_skus(gaps, all_products, min_vertical_overlap)

        # Missing SKU hypotheses (neighbors + image frequency)
        hypotheses = combine_votes_with_freq(votes, all_products, alpha=0.6)

        # Occupancy
        w, h = img.size
        occupancy = occupancy_from_gaps(w, h, gaps)

        return {
            "gap_detected": len(gaps) > 0,
            "gaps": gaps,
            "missing_sku_hypotheses": hypotheses,
            "occupancy_estimate": occupancy,
            "model_version": self.version,
        }


_PIPELINE: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Return a singleton pipeline instance."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = InferencePipeline()
    return _PIPELINE
