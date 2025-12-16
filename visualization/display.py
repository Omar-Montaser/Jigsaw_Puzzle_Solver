"""Display utilities for puzzle visualization."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path


def display_comparison(original: np.ndarray, solved: np.ndarray, 
                       score: Optional[float] = None,
                       title_original: str = "Original (Shuffled)",
                       title_solved: str = "Solved",
                       figsize: tuple = (12, 6)):
    """
    Display original and solved images side by side.
    
    Args:
        original: Original shuffled image
        solved: Solved/reconstructed image
        score: Optional score to display
        title_original: Title for original image
        title_solved: Title for solved image
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert BGR to RGB for display
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original
    
    if len(solved.shape) == 3 and solved.shape[2] == 3:
        solved_rgb = cv2.cvtColor(solved, cv2.COLOR_BGR2RGB)
    else:
        solved_rgb = solved
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(title_original)
    axes[0].axis('off')
    
    solved_title = title_solved
    if score is not None:
        solved_title = f"{title_solved} (Score: {score:.4f})"
    
    axes[1].imshow(solved_rgb)
    axes[1].set_title(solved_title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_comparison(original: np.ndarray, solved: np.ndarray,
                    output_path: str, score: Optional[float] = None,
                    dpi: int = 150):
    """
    Save comparison image to file.
    
    Args:
        original: Original shuffled image
        solved: Solved/reconstructed image
        output_path: Path to save the comparison
        score: Optional score to display
        dpi: Output DPI
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original
    
    if len(solved.shape) == 3 and solved.shape[2] == 3:
        solved_rgb = cv2.cvtColor(solved, cv2.COLOR_BGR2RGB)
    else:
        solved_rgb = solved
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original (Shuffled)")
    axes[0].axis('off')
    
    title = "Solved"
    if score is not None:
        title = f"Solved (Score: {score:.4f})"
    
    axes[1].imshow(solved_rgb)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    output_dir = Path(output_path).parent
    if output_dir and str(output_dir) != '.':
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def display_artifacts(artifacts: Dict[int, dict], piece_ids: Optional[List[int]] = None,
                      figsize_per_piece: tuple = (4, 3)):
    """
    Display artifacts for selected pieces.
    
    Args:
        artifacts: Dict of piece artifacts
        piece_ids: List of piece IDs to display (all if None)
        figsize_per_piece: Figure size per piece
    """
    if piece_ids is None:
        piece_ids = list(artifacts.keys())[:4]  # Show first 4 by default
    
    n_pieces = len(piece_ids)
    fig, axes = plt.subplots(n_pieces, 3, figsize=(figsize_per_piece[0] * 3, 
                                                    figsize_per_piece[1] * n_pieces))
    
    if n_pieces == 1:
        axes = axes.reshape(1, -1)
    
    for i, pid in enumerate(piece_ids):
        artifact = artifacts[pid]
        
        # RGB
        rgb = artifact["rgb"] if isinstance(artifact, dict) else artifact.rgb
        if len(rgb.shape) == 3:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Piece {pid} - RGB")
        axes[i, 0].axis('off')
        
        # Gray
        gray = artifact["gray"] if isinstance(artifact, dict) else artifact.gray
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title(f"Piece {pid} - Enhanced Gray")
        axes[i, 1].axis('off')
        
        # Edges
        edges = artifact["edges"] if isinstance(artifact, dict) else artifact.edges
        axes[i, 2].imshow(edges, cmap='gray')
        axes[i, 2].set_title(f"Piece {pid} - Edges")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_grid(images: List[np.ndarray], grid_size: int,
                 titles: Optional[List[str]] = None,
                 figsize: Optional[tuple] = None):
    """
    Display images in a grid layout.
    
    Args:
        images: List of images to display
        grid_size: Grid dimension (grid_size x grid_size)
        titles: Optional titles for each image
        figsize: Figure size
    """
    if figsize is None:
        figsize = (grid_size * 3, grid_size * 3)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    if grid_size == 1:
        axes = np.array([[axes]])
    
    for idx, img in enumerate(images):
        if idx >= grid_size * grid_size:
            break
        
        r, c = idx // grid_size, idx % grid_size
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[r, c].imshow(img if len(img.shape) == 3 else img, 
                          cmap='gray' if len(img.shape) == 2 else None)
        
        if titles and idx < len(titles):
            axes[r, c].set_title(titles[idx], fontsize=8)
        
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_patches_with_indices(patches: List[np.ndarray], grid_size: int,
                                 figsize: tuple = (8, 8)):
    """
    Display patches in a grid with index labels.
    
    Args:
        patches: List of image patches
        grid_size: Grid dimension
        figsize: Figure size
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    if grid_size == 1:
        axes = np.array([[axes]])
    elif not isinstance(axes, np.ndarray) or axes.ndim == 1:
        axes = np.array(axes).reshape(grid_size, grid_size)
    
    for idx, patch in enumerate(patches):
        row = idx // grid_size
        col = idx % grid_size
        
        ax = axes[row, col]
        
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        
        ax.imshow(patch if len(patch.shape) == 3 else patch,
                  cmap='gray' if len(patch.shape) == 2 else None)
        ax.axis('off')
        ax.set_title(f"Piece {idx}", fontsize=8)
    
    plt.tight_layout()
    plt.show()
