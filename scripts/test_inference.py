from pathlib import Path
from src.wafer.pipeline import predict_image

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"

def main() -> None:
    """
    Test inference on all .png images in the test data directory.
    Prints out the CNN prediction, whether YOLO ran, and number of detections, and timing information.
    """
    images = sorted(DATA_DIR.glob("*.png"))
    if not images:
        raise RuntimeError(f"No .png images found in {DATA_DIR}")

    for img_path in images:
        out = predict_image(img_path.read_bytes())

        print(f"\n=== {img_path.name} ===")

        if not out.get("ok", False):
            err = out.get("error", {}) or {}
            print(
                f"ERROR stage={err.get('stage')} type={err.get('type')} msg={err.get('message')}"
            )
            print("timing_ms:", out.get("timing_ms"))
            continue

        pred = out["cnn"]["pred_class"]
        yolo_ran = out["yolo"]["ran"]
        det_count = len(out["yolo"]["detections"])

        print(f"pred: {pred} | yolo: {yolo_ran} | det: {det_count}")
        print("timing_ms:", out.get("timing_ms"))


if __name__ == "__main__":
    main()
