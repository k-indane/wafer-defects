from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from src.wafer.preprocess import (
    image_bytes_to_cnn_tensor,
    image_bytes_to_yolo_image,
    PreprocessConfig,
)

SPATIAL_CLASSES = {"Loc", "Edge-Loc", "Center", "Scratch", "Donut"}

@dataclass
class ModelBundle:
    """
    Bundle of loaded models and related info.

    Attributes
    ----------
    cnn : tf.keras.Model
        Loaded CNN model.
    classes : List[str]
        List of class names for CNN output.
    yolo : YOLO
        Loaded YOLO model.
    cnn_version : str
        Version string for the CNN model.
    yolo_version : str
        Version string for the YOLO model.
    """
    cnn: tf.keras.Model
    classes: List[str]
    yolo: YOLO
    cnn_version: str = "1.0.0"
    yolo_version: str = "1.0.0"

# Only load models once per process
_MODELS: Optional[ModelBundle] = None

def _repo_root() -> Path:
    """
    Get the repository root directory.

    Returns
    -------
    Path
        Path to the repository root.
    """
    return Path(__file__).resolve().parents[2]

def load_models(
    cnn_path: str = "artifacts/cnn/cnn_model.keras",
    classes_path: str = "artifacts/cnn/classes.json",
    yolo_path: str = "artifacts/yolo/best.pt",
    force_reload: bool = False, # Whether to force reloading the models
) -> ModelBundle:
    global _MODELS
    if _MODELS is not None and not force_reload: # If cached and no reload, return cached models
        return _MODELS

    # Build absolute paths
    root = _repo_root()
    cnn_path_p = (root / cnn_path).resolve()
    classes_path_p = (root / classes_path).resolve()
    yolo_path_p = (root / yolo_path).resolve()

    # Validate files exist
    if not cnn_path_p.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path_p}")
    if not classes_path_p.exists():
        raise FileNotFoundError(f"classes.json not found: {classes_path_p}")
    if not yolo_path_p.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path_p}")

    # Load models
    cnn = tf.keras.models.load_model(str(cnn_path_p))
    classes = json.loads(classes_path_p.read_text())
    if len(classes) != 9:
        raise ValueError(f"Expected 9 classes, got {len(classes)}") # Sanity check
    yolo = YOLO(str(yolo_path_p))
    _MODELS = ModelBundle(cnn=cnn, classes=classes, yolo=yolo) # Cache loaded models
    return _MODELS


def _run_yolo(models: ModelBundle, image_bytes: bytes, cnn_class: str) -> Dict[str, Any]:
    """
    Run YOLO inference on the image bytes.

    Parameters
    ----------
    models : ModelBundle
        Loaded models.
    image_bytes : bytes
        Raw uploaded image bytes.
    cnn_class : str
        Predicted CNN class for routing.
    Returns
    -------
    Dict[str, Any]
        YOLO inference results.
    """
    img = image_bytes_to_yolo_image(image_bytes)  # HxWx3 uint8 output

    # Predict
    results = models.yolo.predict(img, verbose=False)

    # Extract detections
    detections: List[Dict[str, Any]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls_id = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names  # Maps class id to class name
        
        # Build detection dicts for JSON output
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append({
                "cls_id": int(cls_id[i]),
                "cls_name": str(names.get(int(cls_id[i]), int(cls_id[i]))),
                "conf": float(conf[i]),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    # Only keep the highest-confidence box for all classes except Scratch.
    if cnn_class != "Scratch" and len(detections) > 1:
        best = max(detections, key=lambda d: d["conf"])
        detections = [best]

    return {"ran": True, "detections": detections}


def predict_image(image_bytes: bytes, cfg: PreprocessConfig = PreprocessConfig()) -> Dict[str, Any]:
    """
    Run the full prediction pipeline on the given image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw uploaded image bytes.
    cfg : PreprocessConfig
        Preprocessing configuration.
    Returns
    -------
    Dict[str, Any]
        Full inference results including CNN and YOLO outputs.
    """
    
    models = load_models()

    t0 = time.perf_counter()

    # CNN Preprocess
    try:
        xb = image_bytes_to_cnn_tensor(image_bytes, cfg=cfg)
    except Exception as e:
        t1 = time.perf_counter()
        
        # Keep structured response for errors
        return {
            "ok": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "stage": "preprocess",
            },
            "model_versions": {"cnn": getattr(models, "cnn_version", "unknown"), "yolo": getattr(models, "yolo_version", "unknown")},
            "cnn": None,
            "routing": {"should_run_yolo": False, "spatial_classes": sorted(list(SPATIAL_CLASSES))},
            "yolo": {"ran": False, "detections": []},
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "cnn": 0.0,
                "yolo": 0.0,
                "total": (t1 - t0) * 1000.0,
            },
        }

    t1 = time.perf_counter()

    # CNN inference
    try:
        probs = models.cnn.predict(xb, verbose=0)[0].astype(float)
        top_idx = int(np.argmax(probs))
        pred_class = models.classes[top_idx]
    except Exception as e:
        t2 = time.perf_counter()
        return {
            "ok": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "stage": "cnn",
            },
            "model_versions": {"cnn": getattr(models, "cnn_version", "unknown"), "yolo": getattr(models, "yolo_version", "unknown")},
            "cnn": None,
            "routing": {"should_run_yolo": False, "spatial_classes": sorted(list(SPATIAL_CLASSES))},
            "yolo": {"ran": False, "detections": []},
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "cnn": (t2 - t1) * 1000.0,
                "yolo": 0.0,
                "total": (t2 - t0) * 1000.0,
            },
        }

    t2 = time.perf_counter()

    # YOLO routing
    should_run_yolo = pred_class in SPATIAL_CLASSES
    yolo_out: Dict[str, Any] = {"ran": False, "detections": []}
    t3 = t2

    if should_run_yolo:
        try:
            yolo_out = _run_yolo(models, image_bytes, cnn_class=pred_class)
        except Exception as e:
            # Return CNN result with YOLO error
            yolo_out = {
                "ran": False,
                "detections": [],
                "error": {"type": type(e).__name__, "message": str(e)},
            }
        t3 = time.perf_counter()

    # Normal successful return
    return {
        "ok": True,
        "model_versions": {"cnn": models.cnn_version, "yolo": models.yolo_version},
        "cnn": {
            "pred_class": pred_class,
            "top_idx": top_idx,
            "probs": {models.classes[i]: float(probs[i]) for i in range(len(models.classes))},
        },
        "routing": {
            "should_run_yolo": should_run_yolo,
            "spatial_classes": sorted(list(SPATIAL_CLASSES)),
        },
        "yolo": yolo_out,
        "timing_ms": {
            "preprocess": (t1 - t0) * 1000.0,
            "cnn": (t2 - t1) * 1000.0,
            "yolo": (t3 - t2) * 1000.0 if should_run_yolo else 0.0,
            "total": (t3 - t0) * 1000.0,
        },
    }

