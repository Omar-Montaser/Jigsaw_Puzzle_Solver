"""Image enhancement for feature extraction."""

import cv2
import numpy as np


def enhance_grayscale(gray_image, clip_limit=2.0, tile_size=(8, 8),
                      bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                      gaussian_ksize=(3, 3)):
    """
    Enhance grayscale image using CLAHE, bilateral filter, and Gaussian blur.
    
    Args:
        gray_image: Input grayscale image
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
        bilateral_d: Bilateral filter diameter
        bilateral_sigma_color: Bilateral filter sigma for color
        bilateral_sigma_space: Bilateral filter sigma for space
        gaussian_ksize: Gaussian blur kernel size
    
    Returns:
        Enhanced grayscale image
    """
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray_image)
    
    # Bilateral filter preserves edges while smoothing
    enhanced = cv2.bilateralFilter(enhanced, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    
    # Light Gaussian blur for noise reduction
    enhanced = cv2.GaussianBlur(enhanced, gaussian_ksize, 0)
    
    return enhanced


def apply_clahe(gray_image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE contrast enhancement only."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_image)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
