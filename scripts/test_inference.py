from pathlib import Path
import json

from src.wafer.pipeline import predict_image

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"

def main():
    images = sorted([p for p in DATA_DIR.glob("*.png")])
    if not images:
        raise RuntimeError(f"No .png images found in {DATA_DIR}")

    for img_path in images:
        out = predict_image(img_path.read_bytes())
        print(f"\n=== {img_path.name} ===")
        print("pred:", out["cnn"]["pred_class"], "| yolo:", out["yolo"]["ran"], "| det:", len(out["yolo"]["detections"]))

if __name__ == "__main__":
    main()