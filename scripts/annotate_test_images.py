from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Silence warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Silence oneDNN warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from src.wafer.pipeline import predict_image

def _decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
    Decode bytes to an image we can draw on. Handles grayscale wafer maps.
    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes.
    
    Returns
    -------
    np.ndarray
        BGR image array.
    
    Raises
    ------
    ValueError
        If the image cannot be decoded.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode image bytes. Is this a valid PNG/JPG?")
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr


def _draw_detections_boxes_only(img_bgr, dets):
    """
    Draw bounding boxes only on the BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image array to draw on.
    dets : list of dict
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
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return out

def main():
    """
    Annotate test images with YOLO detections and save results.
    1. Loads all .png, .jpg, .jpeg images from the test data
    2. Runs the full prediction pipeline on each image
    3. Draws bounding boxes on the image
    4. Saves the annotated image and JSON result to outputs/annotated/
    5. Prints summary information to console
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data"
    out_dir = repo_root / "outputs" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")))
    if not images:
        raise RuntimeError(f"No images found in {data_dir}")

    for img_path in images:
        image_bytes = img_path.read_bytes()
        result = predict_image(image_bytes)

        # pred = result["cnn"]["pred_class"]
        yolo_ran = result["yolo"]["ran"]
        dets = result["yolo"]["detections"]

        # header = f"CNN: {pred} | YOLO: {yolo_ran} | det: {len(dets)}"

        img_bgr = _decode_image_bytes_to_bgr(image_bytes)
        annotated = _draw_detections_boxes_only(img_bgr, dets)

        out_img = out_dir / f"{img_path.stem}_annotated.png"
        cv2.imwrite(str(out_img), annotated)

        out_json = out_dir / f"{img_path.stem}_result.json"
        out_json.write_text(json.dumps(result, indent=2))

        print(f"Saved: {out_img.name}  (YOLO ran={yolo_ran}, det={len(dets)})")

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
