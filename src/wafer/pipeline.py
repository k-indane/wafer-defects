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
    cnn: tf.keras.Model
    classes: List[str]
    yolo: YOLO
    cnn_version: str = "1.0.0"
    yolo_version: str = "1.0.0"

_MODELS: Optional[ModelBundle] = None

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_models(
    cnn_path: str = "artifacts/cnn/cnn_model.keras",
    classes_path: str = "artifacts/cnn/classes.json",
    yolo_path: str = "artifacts/yolo/best.pt",
    force_reload: bool = False,
) -> ModelBundle:
    global _MODELS
    if _MODELS is not None and not force_reload:
        return _MODELS

    root = _repo_root()
    cnn_path_p = (root / cnn_path).resolve()
    classes_path_p = (root / classes_path).resolve()
    yolo_path_p = (root / yolo_path).resolve()

    if not cnn_path_p.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path_p}")
    if not classes_path_p.exists():
        raise FileNotFoundError(f"classes.json not found: {classes_path_p}")
    if not yolo_path_p.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path_p}")

    cnn = tf.keras.models.load_model(str(cnn_path_p))
    classes = json.loads(classes_path_p.read_text())
    if len(classes) != 9:
        raise ValueError(f"Expected 9 classes, got {len(classes)}")

    yolo = YOLO(str(yolo_path_p))

    _MODELS = ModelBundle(cnn=cnn, classes=classes, yolo=yolo)
    return _MODELS


def _run_yolo(models: ModelBundle, image_bytes: bytes, cnn_class: str) -> Dict[str, Any]:
    img = image_bytes_to_yolo_image(image_bytes)  # HxWx3 uint8

    # Ultralytics returns a list of Results
    results = models.yolo.predict(img, verbose=False)

    detections: List[Dict[str, Any]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls are torch tensors
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls_id = r.boxes.cls.cpu().numpy().astype(int)

        names = r.names  # dict: class_id -> name

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append({
                "cls_id": int(cls_id[i]),
                "cls_name": str(names.get(int(cls_id[i]), int(cls_id[i]))),
                "conf": float(conf[i]),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    # --- Post-processing rule ---
    # Known truth: only keep the single highest-confidence box for all classes except Scratch.
    if cnn_class != "Scratch" and len(detections) > 1:
        best = max(detections, key=lambda d: d["conf"])
        detections = [best]

    return {"ran": True, "detections": detections}


def predict_image(image_bytes: bytes, cfg: PreprocessConfig = PreprocessConfig()) -> Dict[str, Any]:
    models = load_models()

    t0 = time.perf_counter()
    xb = image_bytes_to_cnn_tensor(image_bytes, cfg=cfg)
    t1 = time.perf_counter()

    probs = models.cnn.predict(xb, verbose=0)[0].astype(float)
    top_idx = int(np.argmax(probs))
    pred_class = models.classes[top_idx]
    t2 = time.perf_counter()

    should_run_yolo = pred_class in SPATIAL_CLASSES

    yolo_out = {"ran": False, "detections": []}
    t3 = t2
    if should_run_yolo:
        yolo_out = _run_yolo(models, image_bytes, cnn_class=pred_class)
        t3 = time.perf_counter()

    return {
        "model_versions": {"cnn": models.cnn_version, "yolo": models.yolo_version},
        "cnn": {
            "pred_class": pred_class,
            "top_idx": top_idx,
            "probs": {models.classes[i]: float(probs[i]) for i in range(len(models.classes))}
        },
        "routing": {"should_run_yolo": should_run_yolo, "spatial_classes": sorted(list(SPATIAL_CLASSES))},
        "yolo": yolo_out,
        "timing_ms": {
            "preprocess": (t1 - t0) * 1000.0,
            "cnn": (t2 - t1) * 1000.0,
            "yolo": (t3 - t2) * 1000.0 if should_run_yolo else 0.0,
            "total": (t3 - t0) * 1000.0,
        },
    }
