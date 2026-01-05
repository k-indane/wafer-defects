from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2
import scipy.ndimage


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Tuple[int, int] = (45, 45)
    threshold: int = 127  # for images saved as 0/255
    remove_isolated: bool = True

def image_bytes_to_yolo_image(image_bytes: bytes) -> np.ndarray:
    """
    Returns an HxWx3 uint8 image suitable for Ultralytics YOLO.
    We decode as grayscale and convert to 3-channel so it's robust.
    """
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode image bytes as grayscale")
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr

def resize_wafer_nn(mask: np.ndarray, target_size=(45, 45)) -> np.ndarray:
    """Nearest-neighbor resize (matches your training intent)."""
    zoom_factors = (target_size[0] / mask.shape[0], target_size[1] / mask.shape[1])
    out = scipy.ndimage.zoom(mask, zoom_factors, order=0)
    return out.astype(np.uint8)


def remove_isolated_defects(mask: np.ndarray) -> np.ndarray:
    """
    If center pixel is 1 and all 8 neighbors are 0, drop it to 0.
    (This is the binary version of your 'remove isolated defects' idea.)
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    from scipy.ndimage import generic_filter

    def f(values):
        center = values[4]
        if center == 1:
            neighbors = np.delete(values, 4)
            if np.all(neighbors == 0):
                return 0
        return center

    out = generic_filter(mask, f, size=(3, 3), mode="constant", cval=0)
    return out.astype(np.uint8)


def image_bytes_to_binary_mask(image_bytes: bytes, threshold: int = 127) -> np.ndarray:
    """
    Decode image bytes to a binary 0/1 mask.
    Accepts:
      - true 0/1 images
      - 0/255 images (common when saved as PNG)
    """
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image bytes as grayscale")

    unique = np.unique(img)
    if set(unique.tolist()).issubset({0, 1}):
        return img.astype(np.uint8)

    # Otherwise assume typical grayscale encoding (0..255) and threshold
    mask = (img > threshold).astype(np.uint8)
    return mask


def image_bytes_to_cnn_tensor(image_bytes: bytes, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """
    End-to-end:
      bytes -> binary mask -> isolate removal (optional) -> NN resize -> float32 -> (1,45,45,1)
    """
    mask = image_bytes_to_binary_mask(image_bytes, threshold=cfg.threshold)

    if cfg.remove_isolated:
        mask = remove_isolated_defects(mask)

    mask = resize_wafer_nn(mask, target_size=cfg.target_size)

    x = mask.astype(np.float32)  # keep 0/1 exactly
    x = x[:, :, None]            # (45,45,1)
    x = np.expand_dims(x, 0)     # (1,45,45,1)
    return x
