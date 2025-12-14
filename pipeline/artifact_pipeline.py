"""
Artifact Production Pipeline (Phase 1)

Produces MANDATORY artifacts for puzzle solving:
- rgb: Original BGR image (for reconstruction)
- gray: Enhanced grayscale (CLAHE + bilateral + Gaussian)
- edges: Canny edge map with morphological closing

Artifacts are REQUIRED by all solvers - this is the only entry point.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


def enhance_grayscale(gray: np.ndarray) -> np.ndarray:
    """
    Apply enhancement pipeline to grayscale image.
    
    Pipeline:
    1. CLAHE for contrast enhancement
    2. Bilateral filter for edge-preserving smoothing
    3. Gaussian blur for noise reduction
    """
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Gaussian blur
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges using Canny with morphological closing.
    """
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def create_lowfreq(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """
    Create low-frequency appearance artifact.
    
    Captures global/smooth appearance by heavy blurring.
    This helps match pieces based on overall color/intensity regions
    rather than fine texture details.
    
    Uses both grayscale blur and color blur, combined.
    """
    # Heavy Gaussian blur on grayscale (captures intensity regions)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 5)
    
    # Also blur the color channels (captures color regions)
    bgr_blur = cv2.GaussianBlur(bgr, (15, 15), 5)
    
    # Convert blurred BGR to grayscale for consistency
    bgr_blur_gray = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2GRAY)
    
    # Combine: weighted average of intensity blur and color-derived blur
    # This captures both luminance and chrominance low-frequency info
    lowfreq = cv2.addWeighted(gray_blur, 0.5, bgr_blur_gray, 0.5, 0)
    
    return lowfreq


def create_artifact(bgr_image: np.ndarray) -> dict:
    """
    Create artifact dict from a BGR image.
    
    Args:
        bgr_image: BGR image (numpy array)
    
    Returns:
        dict with 'rgb', 'gray', 'edges', 'blur' keys
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_enhanced = enhance_grayscale(gray)
    edges = detect_edges(gray_enhanced)
    lowfreq = create_lowfreq(gray, bgr_image)
    
    return {
        'rgb': bgr_image.copy(),
        'gray': gray_enhanced,
        'edges': edges,
        'blur': lowfreq  # Low-frequency appearance (PRIMARY)
    }


def split_image_to_pieces(image: np.ndarray, grid_size: int) -> Dict[int, np.ndarray]:
    """Split image into grid_size x grid_size pieces."""
    h, w = image.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    pieces = {}
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            pieces[idx] = image[y1:y2, x1:x2].copy()
            idx += 1
    
    return pieces


def produce_artifacts(image_path: str, grid_size: Optional[int] = None) -> Tuple[Dict[int, dict], int]:
    """
    Produce MANDATORY artifacts from a puzzle image.
    
    This is the ONLY entry point for artifact creation.
    All solvers REQUIRE these artifacts.
    
    Args:
        image_path: Path to the puzzle image
        grid_size: Optional grid size (auto-detected if not provided)
    
    Returns:
        artifacts: Dict[int, dict] where each piece has 'rgb', 'gray', 'edges'
        grid_size: Detected or provided grid size
    
    Raises:
        ValueError: If image cannot be loaded
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Auto-detect grid size if not provided
    if grid_size is None:
        from core.grid_detection import detect_grid_size
        grid_size = detect_grid_size(image_path)
    
    # Split into pieces
    pieces = split_image_to_pieces(img_bgr, grid_size)
    
    # Create artifacts for each piece
    artifacts = {}
    for idx, piece in pieces.items():
        artifacts[idx] = create_artifact(piece)
    
    return artifacts, grid_size


def produce_artifacts_from_pieces(pieces: Dict[int, np.ndarray]) -> Dict[int, dict]:
    """
    Produce artifacts from pre-split pieces.
    
    Args:
        pieces: Dict[int, np.ndarray] - piece_id -> BGR image
    
    Returns:
        artifacts: Dict[int, dict] with 'rgb', 'gray', 'edges' for each piece
    """
    return {idx: create_artifact(piece) for idx, piece in pieces.items()}


def load_and_produce_artifacts(image_path: str, grid_size: Optional[int] = None,
                                verbose: bool = True) -> Tuple[Dict[int, dict], np.ndarray, int]:
    """
    Load image and produce artifacts (convenience function).
    
    Args:
        image_path: Path to the puzzle image
        grid_size: Optional grid size
        verbose: Print progress info
    
    Returns:
        artifacts: Dict of piece artifacts
        original_image: Original loaded image
        grid_size: Grid size used
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if grid_size is None:
        from core.grid_detection import detect_grid_size
        grid_size = detect_grid_size(image_path)
    
    if verbose:
        print(f"Loaded: {image_path}")
        print(f"Size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
        print(f"Grid: {grid_size}x{grid_size}")
    
    artifacts, _ = produce_artifacts(image_path, grid_size)
    
    if verbose:
        print(f"Produced {len(artifacts)} artifacts:")
        print(f"  - rgb: Original BGR ({artifacts[0]['rgb'].shape})")
        print(f"  - gray: Enhanced grayscale ({artifacts[0]['gray'].shape})")
        print(f"  - edges: Canny edges ({artifacts[0]['edges'].shape})")
        print(f"  - blur: Low-frequency appearance ({artifacts[0]['blur'].shape})")
    
    return artifacts, img_bgr, grid_size
