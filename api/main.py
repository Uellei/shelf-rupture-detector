# main.py
import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from .pipeline import get_pipeline
from .schemas import InferenceResponse
from .utils import pil_from_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_pipeline()
    yield


app = FastAPI(title="Shelf Vision API", version="0.2.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "ok",
        "models_loaded_in_ms": pipe.load_ms,
        "version": pipe.version,
    }


def annotate_image(img_bytes: bytes, result: dict) -> bytes:
    img = pil_from_bytes(img_bytes)
    draw = ImageDraw.Draw(img)

    # Gaps = red
    for g in result.get("gaps", []):
        b = g["bbox"]
        x, y = b["x"], b["y"]
        w = b.get("width", b.get("w"))
        h = b.get("height", b.get("h"))
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)

    # Products = green
    for p in result.get("detected_products", []):
        b = p["bbox"]
        x, y = b["x"], b["y"]
        w = b.get("width", b.get("w"))
        h = b.get("height", b.get("h"))
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    image: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    shelf_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
    min_conf_gap: float = Form(0.5),
    min_conf_prod: float = Form(0.5),
    return_image: bool = Form(False),
):
    t0 = time.time()
    img_bytes = await image.read()
    pipe = get_pipeline()
    result = pipe.infer(
        img_bytes, min_conf_gap=min_conf_gap, min_conf_prod=min_conf_prod
    )
    payload = {
        "camera_id": camera_id,
        "shelf_id": shelf_id,
        "timestamp": timestamp,
        "gap_detected": result["gap_detected"],
        "gap_count": len(result["gaps"]),
        "gaps": result["gaps"],
        "missing_sku_hypotheses": result["missing_sku_hypotheses"],
        "occupancy_estimate": result["occupancy_estimate"],
        "latency_ms": int((time.time() - t0) * 1000),
        "model_version": result["model_version"],
        "gap_neighbors": result.get("gap_neighbors", []),
        "image_base64": None,
    }

    if return_image:
        png_bytes = annotate_image(img_bytes, result)
        payload["image_base64"] = "data:image/png;base64," + base64.b64encode(
            png_bytes
        ).decode("utf-8")

    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
