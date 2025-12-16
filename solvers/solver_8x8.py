# solver_8x8.py
"""
8x8 Puzzle Solver - Wrapper for improvements_8x8.py

This module provides a consistent interface for the 8x8 solver,
delegating to the full implementation in improvements_8x8.py.
"""

from typing import Dict, Tuple, List
import numpy as np
import cv2
import tempfile
import os

from .improvements_8x8 import solve_puzzle, SolverConfig


def solve_8x8(artifacts, verbose: bool = True) -> Tuple[Dict[Tuple[int, int], int], Tuple[int, ...], float]:
    """
    Solve an 8x8 jigsaw puzzle.
    
    Args:
        artifacts: Dict or List of dicts, each containing 'rgb' key with piece image.
                   Key/index = piece ID.
        verbose: print progress
    
    Returns:
        board: Dict mapping (row, col) -> piece_id
        arrangement: Flat tuple of piece IDs in row-major order
        score: Final solution score (lower = better)
    """
    # Handle both dict and list input
    if isinstance(artifacts, dict):
        n_pieces = len(artifacts)
        get_piece = lambda idx: artifacts[idx]['rgb']
    else:
        n_pieces = len(artifacts)
        get_piece = lambda idx: artifacts[idx]['rgb']
    
    grid_size = int(np.sqrt(n_pieces))
    
    if grid_size * grid_size != n_pieces:
        raise ValueError(f"Number of pieces ({n_pieces}) is not a perfect square")
    
    # Get piece dimensions from first piece
    sample = get_piece(0)
    ph, pw = sample.shape[:2]
    
    # Reconstruct shuffled puzzle image (pieces in original order)
    h, w = ph * grid_size, pw * grid_size
    puzzle_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx in range(n_pieces):
        r, c = idx // grid_size, idx % grid_size
        puzzle_img[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = get_piece(idx)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, puzzle_img)
    
    try:
        # Solve using improvements_8x8
        config = SolverConfig(verbose=verbose)
        result = solve_puzzle(temp_path, config)
        
        board = result['board']
        arrangement = result['arrangement']
        score = result['score']
        
        return board, arrangement, score
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
