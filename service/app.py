from __future__ import annotations
import os
import warnings
from typing import Any
# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
from src.wafer.pipeline import predict_image

app = FastAPI(
    title="Wafer Defect Detection Service",
    version="1.0.0",
)

# Helper functions
def _read_upload_bytes(file: UploadFile) -> bytes:
    """
    Read and validate uploaded file bytes.

    Parameters
    ----------
    file : UploadFile
        Uploaded file from FastAPI.
    
    Returns
    -------
    bytes
        Raw bytes of the uploaded file.
    
    Raises
    ------
    HTTPException
        If the file is invalid or cannot be read.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. Please upload an image (png/jpg).",
        )
    
    data = file.file.read() # Read bytes

    # Basic size check
    if not data or len(data) < 16:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or too small.")
    return data

def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to BGR format for annotation drawing.
    
    Parameters
    ----------
    image_bytes : bytes
        Raw uploaded image bytes.
    
    Returns
    -------
    np.ndarray
        Decoded BGR image array.
    
    Raises
    ------
    HTTPException
        If the image cannot be decoded.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid PNG/JPG.")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_boxes_only(img_bgr: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes on the image without labels.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image array to draw on.
    dets : list[dict[str, Any]]
        List of detection dictionaries with "xyxy" keys.

    Returns
    -------
    np.ndarray
        Image with bounding boxes drawn.
    """
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)  # thin green
    return out


def _map_pipeline_error_to_http(out: dict[str, Any]) -> None:
    """
    Map pipeline error output to appropriate HTTPException.

    Parameters
    ----------
    out : dict[str, Any]
        Output dictionary from predict_image containing error info.
    
    Raises
    ------
    HTTPException
        Mapped exception based on error stage.
    """
    err = out.get("error") or {}
    stage = err.get("stage", "unknown")
    msg = err.get("message", "Unknown error")

    # Preprocess issues (client)
    if stage == "preprocess":
        raise HTTPException(status_code=400, detail=msg)

    # CNN issues (server)
    if stage == "cnn":
        raise HTTPException(status_code=500, detail=f"CNN inference failed: {msg}")

    # Fallback
    raise HTTPException(status_code=500, detail=msg)


# Endpoints

# Health check
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# Reads image file and runs full pipeline
@app.post("/pipeline")
def pipeline(file: UploadFile = File(...)) -> JSONResponse:
    image_bytes = _read_upload_bytes(file)
    out = predict_image(image_bytes)
    if not out.get("ok", False):
        _map_pipeline_error_to_http(out)
    return JSONResponse(content=out)

# Reads image file, runs pipeline and returns annotated image
@app.post("/pipeline/annotated")
def pipeline_annotated(file: UploadFile = File(...)) -> Response:
    image_bytes = _read_upload_bytes(file)
    out = predict_image(image_bytes)
    if not out.get("ok", False):
        _map_pipeline_error_to_http(out)
    dets = out.get("yolo", {}).get("detections", [])
    img_bgr = _decode_to_bgr(image_bytes)
    annotated = _draw_boxes_only(img_bgr, dets)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated PNG.")
    return Response(content=buf.tobytes(), media_type="image/png")
