import cv2
import numpy as np
from pathlib import Path
from typing import Union

from PIL import Image

# Number of histogram bins per channel (H, S, V). Adjust if you want finer granularity.
_DEFAULT_BINS = (8, 8, 8)


def _load_image(image_or_path: Union[Image.Image, str, Path]) -> Image.Image:
    """
    Accept either a PIL.Image or a filesystem path and return a PIL RGB image.
    """
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
    else:
        image = Image.open(image_or_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Ensure consistent size for all descriptors.
    return image.resize((256, 256))


def extract_color_histogram(
    image_or_path: Union[Image.Image, str, Path], bins=_DEFAULT_BINS
) -> np.ndarray:
    """
    Compute a normalized HSV color histogram for the supplied image.

    Parameters
    ----------
    image_or_path : PIL.Image.Image | str | Path
        Either a PIL image (e.g. the cropped query) or the path to an image file.
    bins : tuple[int, int, int]
        Number of bins for the H, S and V channels.

    Returns
    -------
    np.ndarray shape (1, bins[0] * bins[1] * bins[2])
        Normalized feature vector ready for FAISS consumption.
    """
    image = _load_image(image_or_path)
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    # FAISS expects 2D arrays (n_samples, dim); wrap as batch of size 1.
    return hist.reshape(1, -1)


__all__ = ["extract_color_histogram"]
