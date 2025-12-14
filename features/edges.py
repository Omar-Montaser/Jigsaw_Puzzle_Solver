"""Edge detection for puzzle pieces."""

import cv2
import numpy as np


def detect_edges(gray_image, low_threshold=50, high_threshold=150,
                 morph_kernel_size=3, apply_morphology=True):
    """
    Detect edges using Canny with optional morphological closing.
    
    Args:
        gray_image: Input grayscale image
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
        morph_kernel_size: Morphological kernel size
        apply_morphology: Whether to apply morphological closing
    
    Returns:
        Binary edge image
    """
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    if apply_morphology:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges


def detect_edges_from_rgb(rgb_image, **kwargs):
    """Detect edges from RGB/BGR image."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return detect_edges(gray, **kwargs)


def compute_gradient_magnitude(gray_image):
    """Compute gradient magnitude using Sobel operators."""
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude
