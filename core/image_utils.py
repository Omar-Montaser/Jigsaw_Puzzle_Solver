"""Low-level image operations."""

import cv2
import numpy as np
from PIL import Image


def load_image(file_path):
    """Load image from path and return as numpy array (RGB)."""
    try:
        pic = Image.open(file_path)
        return np.array(pic)
    except Exception as e:
        print(f"File handling error: {e}")
        return None


def load_image_bgr(file_path):
    """Load image using OpenCV (BGR format)."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not load image: {file_path}")
    return img


def to_grayscale(image):
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width=None, height=None):
    """Resize image maintaining aspect ratio if only one dimension given."""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        ratio = width / w
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
