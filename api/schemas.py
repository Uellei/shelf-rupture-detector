from typing import List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DetectedProduct(BaseModel):
    sku_id: str
    bbox: BBox
    score: float


class GapNeighbors(BaseModel):
    left: Optional[DetectedProduct] = None
    right: Optional[DetectedProduct] = None


class Gap(BaseModel):
    bbox: BBox
    final_score: float
    area_px: int
    aspect_ratio: float
    neighbors: Optional[GapNeighbors] = None


class SKUHypothesis(BaseModel):
    sku_id: str
    confidence: float


class InferenceResponse(BaseModel):
    image_base64: Optional[str] = None
    camera_id: Optional[str] = None
    shelf_id: Optional[str] = None
    timestamp: Optional[str] = None
    gap_detected: bool
    gap_count: int
    gaps: List[Gap]
    missing_sku_hypotheses: List[SKUHypothesis]
    occupancy_estimate: float = Field(ge=0, le=1)
    latency_ms: int
    model_version: str
