"""Artifact data model for puzzle pieces."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .enhancement import enhance_grayscale
from .edges import detect_edges


@dataclass
class PieceArtifact:
    """
    Structured artifact for a single puzzle piece.
    
    Attributes:
        rgb: Original BGR image (for reconstruction)
        gray: Enhanced grayscale (for texture matching)
        edges: Canny edge map (for edge continuity matching)
    """
    rgb: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    
    @classmethod
    def from_bgr(cls, bgr_image: np.ndarray) -> 'PieceArtifact':
        """Create artifact from BGR image."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_enhanced = enhance_grayscale(gray)
        edges = detect_edges(gray_enhanced)
        
        return cls(rgb=bgr_image, gray=gray_enhanced, edges=edges)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "rgb": self.rgb,
            "gray": self.gray,
            "edges": self.edges
        }


def create_artifacts(image_path: str) -> tuple:
    """
    Process an image and produce structured artifacts for each puzzle piece.
    
    Args:
        image_path: Path to the puzzle image
        
    Returns:
        artifacts: Dict[int, PieceArtifact]
        grid_size: Detected grid size (2, 4, or 8)
    """
    from core.grid_detection import detect_grid_size
    from core.splitting import split_image
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    grid_size = detect_grid_size(image_path)
    patches = split_image(img_bgr, grid_size)
    
    artifacts = {}
    for idx, patch in enumerate(patches):
        artifacts[idx] = PieceArtifact.from_bgr(patch)
    
    return artifacts, grid_size


def create_artifacts_from_pieces(pieces: Dict[int, np.ndarray]) -> Dict[int, PieceArtifact]:
    """
    Process pre-split pieces into artifacts.
    
    Args:
        pieces: Dict[int, np.ndarray] - piece_id -> BGR image
        
    Returns:
        artifacts: Dict[int, PieceArtifact]
    """
    return {idx: PieceArtifact.from_bgr(patch) for idx, patch in pieces.items()}


# Backward compatibility: dict-based interface
def create_artifacts_dict(image_path: str) -> tuple:
    """Create artifacts as dictionaries (backward compatible)."""
    artifacts, grid_size = create_artifacts(image_path)
    return {idx: art.to_dict() for idx, art in artifacts.items()}, grid_size


def create_artifacts_from_pieces_dict(pieces: Dict[int, np.ndarray]) -> Dict[int, dict]:
    """Create artifacts from pieces as dictionaries (backward compatible)."""
    artifacts = create_artifacts_from_pieces(pieces)
    return {idx: art.to_dict() for idx, art in artifacts.items()}
