"""
2x2 Puzzle Solver - RGB-Only Implementation

Simple RGB-based matching for 2x2 puzzles.
Uses SSD + NCC + continuity scoring on raw RGB pixels.

Algorithm:
- Exhaustive search over all 24 permutations
- Swap hillclimb refinement (rarely changes anything)
"""

import random
import numpy as np
from itertools import permutations
from typing import Dict, Tuple


def extract_edge(piece: np.ndarray, edge: str, strip_width: int = 10) -> np.ndarray:
    """Extract edge strip from a piece."""
    if edge == 'top':
        return piece[:strip_width, :, :].astype(np.float32)
    elif edge == 'bottom':
        return piece[-strip_width:, :, :].astype(np.float32)
    elif edge == 'left':
        return piece[:, :strip_width, :].astype(np.float32)
    elif edge == 'right':
        return piece[:, -strip_width:, :].astype(np.float32)
    else:
        raise ValueError(f"Unknown edge: {edge}")


def seam_cost(pieces: Dict[int, np.ndarray], A: int, edgeA: str, 
              B: int, edgeB: str) -> float:
    """
    Compute seam cost between two piece edges using SSD, NCC, and continuity.
    
    Args:
        pieces: dict of piece_id -> piece image (RGB)
        A: piece id for first piece
        edgeA: edge of piece A ('top', 'bottom', 'left', 'right')
        B: piece id for second piece
        edgeB: edge of piece B
    
    Returns:
        Combined score (lower is better match)
    """
    stripA = extract_edge(pieces[A], edgeA, strip_width=10)
    stripB = extract_edge(pieces[B], edgeB, strip_width=10)
    
    # Get boundary pixels and near-boundary strips
    if edgeA == 'right' and edgeB == 'left':
        edgeA_pixels = stripA[:, -1, :].flatten()
        edgeB_pixels = stripB[:, 0, :].flatten()
        stripA_near = stripA[:, -3:, :]
        stripB_near = stripB[:, :3, :]
    elif edgeA == 'bottom' and edgeB == 'top':
        edgeA_pixels = stripA[-1, :, :].flatten()
        edgeB_pixels = stripB[0, :, :].flatten()
        stripA_near = stripA[-3:, :, :]
        stripB_near = stripB[:3, :, :]
    else:
        edgeA_pixels = stripA.flatten()
        edgeB_pixels = stripB.flatten()
        stripA_near = stripA
        stripB_near = stripB
    
    # 1. SSD - Sum of Squared Differences
    ssd_score = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    
    # 2. Normalized Cross-Correlation
    mean_a = np.mean(edgeA_pixels)
    mean_b = np.mean(edgeB_pixels)
    std_a = np.std(edgeA_pixels) + 1e-10
    std_b = np.std(edgeB_pixels) + 1e-10
    ncc = np.mean((edgeA_pixels - mean_a) * (edgeB_pixels - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100
    
    # 3. Gradient continuity
    var_a = np.var(stripA_near.flatten()) + 1e-10
    var_b = np.var(stripB_near.flatten()) + 1e-10
    seam_diff = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    continuity_score = seam_diff / ((var_a + var_b) / 2)
    
    return 0.3 * np.sqrt(ssd_score) + 0.3 * ncc_cost + 0.4 * continuity_score * 10


def compute_puzzle_score(pieces: Dict[int, np.ndarray], arrangement: list) -> float:
    """
    Compute total puzzle score for a 2x2 arrangement.
    
    Arrangement layout:
        P[0]  P[1]
        P[2]  P[3]
    """
    P = arrangement
    score = 0.0
    # Horizontal seams
    score += seam_cost(pieces, P[0], 'right', P[1], 'left')
    score += seam_cost(pieces, P[2], 'right', P[3], 'left')
    # Vertical seams
    score += seam_cost(pieces, P[0], 'bottom', P[2], 'top')
    score += seam_cost(pieces, P[1], 'bottom', P[3], 'top')
    return score


def exhaustive_search(pieces: Dict[int, np.ndarray]) -> Tuple[list, float]:
    """Try all 24 permutations and find the one with lowest score."""
    piece_ids = list(pieces.keys())
    best_score = float('inf')
    best_arrangement = None
    
    for perm in permutations(piece_ids):
        arrangement = list(perm)
        score = compute_puzzle_score(pieces, arrangement)
        if score < best_score:
            best_score = score
            best_arrangement = arrangement
    
    return best_arrangement, best_score


def swap_hillclimb(pieces: Dict[int, np.ndarray], arrangement: list,
                   max_iterations: int = 1000) -> Tuple[list, float]:
    """Swap hillclimb refinement."""
    current = arrangement.copy()
    current_score = compute_puzzle_score(pieces, current)
    
    for _ in range(max_iterations):
        i, j = random.sample(range(4), 2)
        current[i], current[j] = current[j], current[i]
        new_score = compute_puzzle_score(pieces, current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            current[i], current[j] = current[j], current[i]
    
    return current, current_score


def solve_2x2(artifacts: Dict[int, dict], verbose: bool = True) -> Tuple[list, float]:
    """
    Main solver function for 2x2 puzzles.
    
    Uses RGB from artifacts for simple SSD+NCC+continuity matching.
    
    Args:
        artifacts: dict mapping piece_id (0-3) to artifact dict
                   Each artifact MUST contain: 'rgb'
        verbose: print progress info
    
    Returns:
        arrangement: list of 4 piece ids in solved order
        score: final puzzle score
    """
    # Extract RGB pieces from artifacts
    pieces = {pid: art['rgb'] for pid, art in artifacts.items()}
    
    if verbose:
        print("=" * 50)
        print("2x2 Puzzle Solver (RGB-Only)")
        print("=" * 50)
        print(f"Pieces: {len(pieces)}")
    
    # Exhaustive search
    if verbose:
        print("\nExhaustive search over 24 permutations...")
    
    arrangement, score = exhaustive_search(pieces)
    
    if verbose:
        print(f"  Best arrangement: {arrangement}")
        print(f"  Best score: {score:.4f}")
    
    # Swap hillclimb
    if verbose:
        print("\nSwap hillclimb refinement...")
    
    arrangement, score = swap_hillclimb(pieces, arrangement, 1000)
    
    if verbose:
        print(f"  Final arrangement: {arrangement}")
        print(f"  Final score: {score:.4f}")
    
    return arrangement, score


def arrangement_to_board(arrangement: list, grid_size: int = 2) -> dict:
    """Convert flat arrangement to board dict format."""
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board
