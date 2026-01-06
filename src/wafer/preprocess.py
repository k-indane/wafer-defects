from dataclasses import dataclass
from typing import Tuple
from io import BytesIO
import numpy as np
from PIL import Image

# Exceptions
class PreprocessError(Exception):
    """
    Base class for preprocessing + input errors.
    Mapped to HTTP 400 Bad Request in API.
    """

class InvalidImageError(PreprocessError):
    """
    Raised when input bytes are not a valid decodable image.
    Examples:
    - Empty/corrupted file
    - Unsupported format (not PNG/JPG)
    """

class InvalidImageShapeError(PreprocessError):
    """
    Raised when image has unsupported dimensions or shape
    Examples:
    - Too small/large
    - Wrong number of channels or dimensions
    """

# Config
@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration object for preprocessing steps.
    Ensures assumptions made during model training are enforced at inference.
    """
    target_size: Tuple[int, int] = (45, 45) # 45x45 input
    force_grayscale: bool = True            # Convert RGB to grayscale
    binarize: bool = True                   # Enforce 1 or 0
    binarize_threshold: int = 127           # Threshold for 1 or 0
    allowed_min_side: int = 10              # Reject small images
    allowed_max_side: int = 4096            # Reject large images

# Helper functions
def _load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Decode raw image bytes to a PIL image.
    
    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file.

    Returns
    -------
    PIL.Image.Image
        Decoded image object.

    Raises
    ------
    InvalidImageError
        If the bytes are empty or cannot be decoded as an image.
    """
    if not image_bytes or len(image_bytes) < 16:
        raise InvalidImageError("Uploaded file is empty or too small.")
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()  # Force full decode
        return img
    except Exception as e:
        raise InvalidImageError(
            "Could not decode image bytes. Only valid PNG/JPG files are accepted."
        ) from e

def _validate_image_size(img: Image.Image, cfg: PreprocessConfig) -> None:
    """
    Validate that image dimensions are within limits.
    
    Parameters
    ----------
    img : PIL.Image.Image
        Decoded image object.
    cfg : PreprocessConfig
        Preprocessing configuration.
    
    Raises
    ------
    InvalidImageShapeError
        If image dimensions are too small or too large.
    """
    w, h = img.size
    if w < cfg.allowed_min_side or h < cfg.allowed_min_side:
        raise InvalidImageShapeError(f"Image is too small ({w}x{h}).")
    if w > cfg.allowed_max_side or h > cfg.allowed_max_side:
        raise InvalidImageShapeError(f"Image is too large ({w}x{h}).")

def _to_grayscale(img: Image.Image) -> Image.Image:
    """
    Convert image to grayscale.

    Parameters
    ----------
    img : PIL.Image.Image
        Decoded image object (in any color mode).
    
    Returns
    -------
    PIL.Image.Image
        Grayscale image ("L" mode).
    """
    return img.convert("L")

def _resize_nearest(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image using nearest neighbor interpolation.

    Parameters
    ----------
    img : PIL.Image.Image
        Decoded image object.
    target_size : Tuple[int, int]
        Desired output size (width, height).

    Returns
    -------
    PIL.Image.Image
        Resized image.
    """
    return img.resize(target_size, resample=Image.Resampling.NEAREST)

def _binarize_uint8(gray_u8: np.ndarray, threshold: int) -> np.ndarray:
    """
    Binarize a grayscale uint8 image into 0s and 1s.

    Parameters
    ----------
    gray_u8 : np.ndarray
        Grayscale image as a 2D uint8 array.
    threshold : int
        Pixel values greater than this threshold (127) are set to 1.

    Returns
    -------
    np.ndarray
        Binary image array with values 0 or 1.
    """
    return (gray_u8 > threshold).astype(np.uint8)

def _ensure_2d(gray_u8: np.ndarray) -> np.ndarray:
    """
    Ensure the grayscale image array is 2D.

    Parameters
    ----------
    gray_u8 : np.ndarray
        Grayscale image array.
    
    Returns
    -------
    np.ndarray
        2D grayscale image array.
    
    Raises
    ------
    InvalidImageShapeError
        If the input array is not 2D or single-channel.
    """
    if gray_u8.ndim == 2:
        return gray_u8
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 1:
        return gray_u8[:, :, 0]
    raise InvalidImageShapeError(f"Expected grayscale image, got shape {gray_u8.shape}.")

# Main functions
def image_bytes_to_cnn_tensor(image_bytes: bytes, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """
    Preprocess image bytes into CNN input tensor.
    The output tensor matches the shape and format used during model training.
    - shape: (1, 45, 45, 1)
    - dtype: float32
    - values: [0.0, 1.0] or {0.0, 1.0} if binarized

    Parameters
    ----------
    image_bytes : bytes
        Raw uploaded image bytes.
    cfg : PreprocessConfig
        Preprocessing configuration.
    
    Returns
    -------
    np.ndarray
        CNN input tensor.

    Raises
    ------
    PreprocessError
        If any preprocessing step fails.
    """
    img = _load_image_from_bytes(image_bytes)
    _validate_image_size(img, cfg)
    if cfg.force_grayscale:
        img = _to_grayscale(img)
    img = _resize_nearest(img, cfg.target_size)
    gray_u8 = np.array(img, dtype=np.uint8)
    gray_u8 = _ensure_2d(gray_u8)
    if cfg.binarize:
        bin_u8 = _binarize_uint8(gray_u8, cfg.binarize_threshold)
        x = bin_u8.astype(np.float32)
    else:
        x = (gray_u8.astype(np.float32) / 255.0)
    x = x.reshape((1, x.shape[0], x.shape[1], 1)) # (1, 45, 45, 1)
    return x

def image_bytes_to_yolo_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image bytes into YOLO input image array.
    The output image matches the shape and format used during YOLO model training.
    - shape: (H, W, 3)
    - dtype: uint8
    - values: [0, 255]

    Parameters
    ----------
    image_bytes : bytes
        Raw uploaded image bytes.

    Returns
    -------
    np.ndarray
        YOLO input image array.
    
    Raises
    ------
    PreprocessError
        If any preprocessing step fails.
    """
    img = _load_image_from_bytes(image_bytes)
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8) # (H, W, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InvalidImageShapeError(f"Expected 3-channel RGB image for YOLO, got shape {arr.shape}.")
    return arr
