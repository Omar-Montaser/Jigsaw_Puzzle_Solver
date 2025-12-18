"""
Shared accuracy testing utilities for puzzle solver evaluation.

This module provides:
- Pairwise Neighbor Accuracy: measures relative adjacency correctness
- Aligned Accuracy: accounts for global transforms (shifts, flips)
- Ground truth loading from correct images
- Helper functions for grid manipulation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


# =============================================================================
# PAIRWISE NEIGHBOR ACCURACY
# =============================================================================

def compute_pairwise_neighbor_accuracy(
    reconstructed_grid: np.ndarray,
    ground_truth_labels: Dict[int, Tuple[int, int]]
) -> float:
    """
    Compute Pairwise Neighbor Accuracy for a reconstructed puzzle grid.
    This implementation is intentionally identical to the notebook evaluation
    logic (AAA.ipynb, AAA_4x4.ipynb) for reproducibility.
    
    Algorithm:
        1. Count only RIGHT and BOTTOM adjacency pairs (not bidirectional)
        2. Skip pairs containing -1 (empty cells)
        3. Use ground-truth right and bottom neighbors only
        4. Accuracy = correct_pairs / total_pairs
        5. Return 0.0 if total_pairs == 0
    
    Args:
        reconstructed_grid: 2D numpy array of piece IDs, with -1 for empty cells.
        ground_truth_labels: Dict mapping piece_id -> (row, col) in the solved puzzle.
    
    Returns:
        Score in [0.0, 1.0] representing the fraction of correct neighbor pairs.
    
    Example (2x2 grid):
        # Ground truth: piece 0 at (0,0), piece 1 at (0,1), piece 2 at (1,0), piece 3 at (1,1)
        gt = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        # Perfect reconstruction: [[0,1],[2,3]]
        grid_perfect = np.array([[0,1],[2,3]])
        # -> accuracy = 1.0 (4 pairs: 0-1, 2-3 horizontal; 0-2, 1-3 vertical, all correct)
        
        # One wrong neighbor: [[0,2],[1,3]] (swapped 1 and 2)
        grid_wrong = np.array([[0,2],[1,3]])
        # -> accuracy = 0.25 (only 1-3 vertical is correct, 3 pairs wrong)
    """
    rows, cols = reconstructed_grid.shape
    
    # Build ground-truth neighbor sets from labels
    pos_to_piece = {pos: pid for pid, pos in ground_truth_labels.items()}
    
    gt_right_neighbors = {}
    gt_bottom_neighbors = {}
    
    for piece_id, (r, c) in ground_truth_labels.items():
        gt_right_neighbors[piece_id] = pos_to_piece.get((r, c + 1))
        gt_bottom_neighbors[piece_id] = pos_to_piece.get((r + 1, c))
    
    correct_pairs = 0
    total_pairs = 0
    
    # Check horizontal pairs
    for r in range(rows):
        for c in range(cols - 1):
            left_piece = reconstructed_grid[r, c]
            right_piece = reconstructed_grid[r, c + 1]
            
            if left_piece == -1 or right_piece == -1:
                continue
            
            total_pairs += 1
            if gt_right_neighbors.get(left_piece) == right_piece:
                correct_pairs += 1
    
    # Check vertical pairs
    for r in range(rows - 1):
        for c in range(cols):
            top_piece = reconstructed_grid[r, c]
            bottom_piece = reconstructed_grid[r + 1, c]
            
            if top_piece == -1 or bottom_piece == -1:
                continue
            
            total_pairs += 1
            if gt_bottom_neighbors.get(top_piece) == bottom_piece:
                correct_pairs += 1
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0


# =============================================================================
# ALIGNED ACCURACY
# =============================================================================

def aligned_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute alignment-aware accuracy between predicted and ground truth grids.
    
    Args:
        pred: (N, N) grid of piece IDs (predicted arrangement)
        gt: (N, N) grid of piece IDs (ground truth arrangement)
    
    Returns:
        Maximum fraction of matching tiles after applying global transforms.
        
    Allowed transforms:
        - Cyclic row shifts (0 to N-1)
        - Cyclic column shifts (0 to N-1)
        - Optional horizontal flip
        - Optional vertical flip
    """
    n = pred.shape[0]
    total = n * n
    best_accuracy = 0.0
    
    # Generate all flip variants of pred
    flip_variants = [
        pred,                          # No flip
        np.fliplr(pred),               # Horizontal flip
        np.flipud(pred),               # Vertical flip
        np.flipud(np.fliplr(pred)),    # Both flips
    ]
    
    for flipped in flip_variants:
        # Try all cyclic shifts
        for row_shift in range(n):
            for col_shift in range(n):
                # Apply cyclic shifts
                shifted = np.roll(flipped, row_shift, axis=0)
                shifted = np.roll(shifted, col_shift, axis=1)
                
                # Count matches
                matches = np.sum(shifted == gt)
                accuracy = matches / total
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    
                # Early exit if perfect
                if best_accuracy == 1.0:
                    return 1.0
    
    return best_accuracy


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_pieces(image: np.ndarray, grid_size: int) -> List[np.ndarray]:
    """Extract pieces from an image in row-major order."""
    h, w = image.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    pieces = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            pieces.append(image[y1:y2, x1:x2].copy())
    return pieces


def load_ground_truth(
    puzzle_path: str,
    correct_path: str,
    grid_size: int
) -> Dict[int, Tuple[int, int]]:
    """
    Derive ground truth labels by matching shuffled pieces to the correct image.
    
    Args:
        puzzle_path: Path to shuffled puzzle image
        correct_path: Path to correct (unshuffled) image
        grid_size: Size of the puzzle grid (e.g., 2, 4, 8)
    
    Returns:
        Dict mapping piece_id -> (row, col) in the solved puzzle
    """
    puzzle_img = cv2.imread(puzzle_path)
    correct_img = cv2.imread(correct_path)
    
    if puzzle_img is None:
        raise ValueError(f"Could not load puzzle image: {puzzle_path}")
    if correct_img is None:
        raise ValueError(f"Could not load correct image: {correct_path}")
    
    puzzle_pieces = extract_pieces(puzzle_img, grid_size)
    correct_pieces = extract_pieces(correct_img, grid_size)
    
    # Compute all match scores
    scores = []
    for piece_id, piece in enumerate(puzzle_pieces):
        for pos_idx, correct_piece in enumerate(correct_pieces):
            if correct_piece.shape != piece.shape:
                correct_piece = cv2.resize(correct_piece, (piece.shape[1], piece.shape[0]))
            diff = np.sum((piece.astype(float) - correct_piece.astype(float)) ** 2)
            scores.append((diff, piece_id, pos_idx))
    
    # Greedy assignment
    scores.sort()
    ground_truth_labels = {}
    assigned_pieces = set()
    used_positions = set()
    
    for diff, piece_id, pos_idx in scores:
        if piece_id in assigned_pieces or pos_idx in used_positions:
            continue
        
        row, col = pos_idx // grid_size, pos_idx % grid_size
        ground_truth_labels[piece_id] = (row, col)
        assigned_pieces.add(piece_id)
        used_positions.add(pos_idx)
        
        if len(assigned_pieces) == len(puzzle_pieces):
            break
    
    return ground_truth_labels


def load_puzzle_with_ground_truth(
    puzzle_path: str,
    correct_path: str,
    grid_size: int,
    artifact_creator=None
) -> Tuple[dict, Dict[int, Tuple[int, int]]]:
    """
    Load shuffled puzzle pieces and derive ground truth labels by matching
    to the correct (unshuffled) image.
    
    Args:
        puzzle_path: Path to shuffled puzzle image
        correct_path: Path to correct (unshuffled) image
        grid_size: Size of the puzzle grid (e.g., 2, 4, 8)
        artifact_creator: Optional function to create artifacts (e.g., create_artifact from pipeline)
                         If None, creates simple {'rgb': piece} dict
    
    Returns:
        Tuple of (artifacts dict, ground_truth_labels dict)
    """
    puzzle_img = cv2.imread(puzzle_path)
    correct_img = cv2.imread(correct_path)
    
    if puzzle_img is None:
        raise ValueError(f"Could not load puzzle image: {puzzle_path}")
    if correct_img is None:
        raise ValueError(f"Could not load correct image: {correct_path}")
    
    puzzle_pieces = extract_pieces(puzzle_img, grid_size)
    correct_pieces = extract_pieces(correct_img, grid_size)
    
    # Build artifacts dict
    artifacts = {}
    for idx, piece in enumerate(puzzle_pieces):
        if artifact_creator is not None:
            artifacts[idx] = artifact_creator(piece)
        else:
            artifacts[idx] = {'rgb': piece}
    
    # Compute all match scores
    scores = []
    for piece_id, piece in enumerate(puzzle_pieces):
        for pos_idx, correct_piece in enumerate(correct_pieces):
            if correct_piece.shape != piece.shape:
                correct_piece = cv2.resize(correct_piece, (piece.shape[1], piece.shape[0]))
            diff = np.sum((piece.astype(float) - correct_piece.astype(float)) ** 2)
            scores.append((diff, piece_id, pos_idx))
    
    # Greedy assignment
    scores.sort()
    ground_truth_labels = {}
    assigned_pieces = set()
    used_positions = set()
    
    for diff, piece_id, pos_idx in scores:
        if piece_id in assigned_pieces or pos_idx in used_positions:
            continue
        
        row, col = pos_idx // grid_size, pos_idx % grid_size
        ground_truth_labels[piece_id] = (row, col)
        assigned_pieces.add(piece_id)
        used_positions.add(pos_idx)
        
        if len(assigned_pieces) == len(puzzle_pieces):
            break
    
    return artifacts, ground_truth_labels


def gt_labels_to_grid(
    gt_labels: Dict[int, Tuple[int, int]], 
    grid_size: int
) -> np.ndarray:
    """Convert ground truth labels dict to 2D grid array."""
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    for piece_id, (row, col) in gt_labels.items():
        grid[row, col] = piece_id
    return grid


def arrangement_to_grid(arrangement, grid_size: int) -> np.ndarray:
    """Convert flat arrangement (list or tuple) to 2D grid array."""
    return np.array(arrangement).reshape(grid_size, grid_size)


def reconstruct_image(
    artifacts: dict, 
    arrangement, 
    grid_size: int
) -> np.ndarray:
    """Reconstruct the solved puzzle image from arrangement."""
    # Get piece dimensions from first artifact
    sample = artifacts[0]['rgb'] if 'rgb' in artifacts[0] else artifacts[0]
    piece_h, piece_w = sample.shape[:2]
    
    output = np.zeros((piece_h * grid_size, piece_w * grid_size, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = arrangement[idx]
            if piece_id != -1:
                y1, y2 = r * piece_h, (r + 1) * piece_h
                x1, x2 = c * piece_w, (c + 1) * piece_w
                piece = artifacts[piece_id]['rgb'] if 'rgb' in artifacts[piece_id] else artifacts[piece_id]
                output[y1:y2, x1:x2] = piece
            idx += 1
    
    return output