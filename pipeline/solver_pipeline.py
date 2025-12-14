"""
Solver Pipeline (Phase 2)

Orchestrates puzzle solving using MANDATORY Phase 1 artifacts.
Artifacts are produced first, then consumed by solvers.

This pipeline enforces artifact-first architecture:
1. Load image → produce artifacts (MANDATORY)
2. Solve using artifacts (edges + texture + RGB)
3. Reconstruct using RGB from artifacts
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from .artifact_pipeline import produce_artifacts, load_and_produce_artifacts


def solve_puzzle(artifacts: Dict[int, dict], grid_size: int,
                 verbose: bool = True) -> Tuple[dict, list, float]:
    """
    Solve a puzzle using MANDATORY artifacts.
    
    Automatically selects the appropriate solver based on grid size.
    
    Args:
        artifacts: Dict of piece artifacts (MANDATORY)
                   Each must contain: 'rgb', 'gray', 'edges'
        grid_size: Size of the puzzle grid
        verbose: Print progress info
    
    Returns:
        board: Dict mapping (row, col) -> piece_id
        arrangement: Flat list of piece ids
        score: Final puzzle score
    
    Raises:
        ValueError: If artifacts are missing or invalid
    """
    if grid_size == 2:
        from solvers.solver_2x2 import solve_2x2, arrangement_to_board
        arrangement, score = solve_2x2(artifacts, verbose=verbose)
        board = arrangement_to_board(arrangement, grid_size=2)
        return board, arrangement, score
    
    elif grid_size == 4:
        from solvers.solver_4x4 import solve_4x4
        return solve_4x4(artifacts, verbose=verbose)
    
    elif grid_size == 8:
        from solvers.solver_8x8 import solve_8x8
        return solve_8x8(artifacts, verbose=verbose)
    
    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")


def reconstruct_image(artifacts: Dict[int, dict], board: dict,
                      grid_size: int, show_numbers: bool = False) -> np.ndarray:
    """
    Reconstruct the solved puzzle image from artifacts.
    
    Uses RGB channel from artifacts for reconstruction.
    
    Args:
        artifacts: Dict of piece artifacts
        board: Dict mapping (row, col) -> piece_id
        grid_size: Size of the puzzle grid
        show_numbers: Overlay piece numbers on output
    
    Returns:
        Reconstructed image
    """
    sample_rgb = artifacts[list(artifacts.keys())[0]]['rgb']
    piece_h, piece_w = sample_rgb.shape[:2]
    
    output_h = piece_h * grid_size
    output_w = piece_w * grid_size
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = board[(r, c)]
            y1, y2 = r * piece_h, (r + 1) * piece_h
            x1, x2 = c * piece_w, (c + 1) * piece_w
            output[y1:y2, x1:x2] = artifacts[piece_id]['rgb']
    
    if show_numbers:
        for r in range(grid_size):
            for c in range(grid_size):
                piece_id = board[(r, c)]
                center_x = c * piece_w + piece_w // 2
                center_y = r * piece_h + piece_h // 2
                
                text = str(piece_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = min(piece_w, piece_h) / 80.0
                thickness = max(2, int(font_scale * 2))
                
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
                
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return output


def solve_and_reconstruct(image_path: str, output_path: Optional[str] = None,
                          grid_size: Optional[int] = None,
                          verbose: bool = True) -> Tuple[list, float, np.ndarray]:
    """
    Complete pipeline: load → produce artifacts → solve → reconstruct.
    
    This is the main entry point that enforces artifact-first architecture.
    
    Args:
        image_path: Path to puzzle image
        output_path: Optional path to save result
        grid_size: Optional grid size (auto-detected if not provided)
        verbose: Print progress info
    
    Returns:
        arrangement: Flat list of piece ids
        score: Final puzzle score
        solved_image: Reconstructed image
    """
    # Phase 1: Produce artifacts (MANDATORY)
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Artifact Production")
        print("=" * 60)
    
    artifacts, original, grid_size = load_and_produce_artifacts(image_path, grid_size, verbose)
    
    # Phase 2: Solve using artifacts
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Puzzle Solving (Artifact-First)")
        print("=" * 60)
    
    board, arrangement, score = solve_puzzle(artifacts, grid_size, verbose)
    
    # Reconstruct using RGB from artifacts
    solved = reconstruct_image(artifacts, board, grid_size)
    
    # Save if requested
    if output_path:
        output_dir = Path(output_path).parent
        if output_dir and str(output_dir) != '.':
            output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, solved)
        if verbose:
            print(f"\nSaved: {output_path}")
    
    return arrangement, score, solved


def solve_image(image_path: str, output_path: Optional[str] = None,
                verbose: bool = True) -> Tuple[list, float, np.ndarray]:
    """Convenience alias for solve_and_reconstruct."""
    return solve_and_reconstruct(image_path, output_path, verbose=verbose)
