"""
2x2 Puzzle Solver - Artifact-First Implementation

REQUIRES Phase 1 artifacts (edges, enhanced grayscale).
Uses multi-feature matching: edges (primary) + texture (primary) + RGB (secondary).

Algorithm:
- Phase 1: Multi-feature seam cost using artifacts
- Phase 2: Exhaustive search over all 24 permutations
- Phase 3: Optional swap hillclimb refinement
"""

import random
from itertools import permutations
from typing import Dict, Tuple

from .seam_cost import seam_cost, validate_artifacts, SeamCostWeights


def compute_puzzle_score(artifacts: Dict[int, dict], arrangement: list,
                         weights: SeamCostWeights = None) -> float:
    """
    Compute total puzzle score for a 2x2 arrangement using artifacts.
    
    Arrangement layout:
        P[0]  P[1]
        P[2]  P[3]
    
    Args:
        artifacts: dict of piece_id -> artifact dict (MANDATORY)
        arrangement: list of 4 piece ids
        weights: feature weights
    
    Returns:
        Total seam cost (lower is better)
    """
    P = arrangement
    
    score = 0.0
    # Horizontal seams
    score += seam_cost(artifacts, P[0], 'right', P[1], 'left', weights)
    score += seam_cost(artifacts, P[2], 'right', P[3], 'left', weights)
    # Vertical seams
    score += seam_cost(artifacts, P[0], 'bottom', P[2], 'top', weights)
    score += seam_cost(artifacts, P[1], 'bottom', P[3], 'top', weights)
    
    return score


def exhaustive_search(artifacts: Dict[int, dict], 
                      weights: SeamCostWeights = None) -> Tuple[list, float]:
    """
    Try all 24 permutations and find the one with lowest score.
    
    Args:
        artifacts: dict of piece_id -> artifact dict (MANDATORY)
        weights: feature weights
    
    Returns:
        best_arrangement: list of 4 piece ids
        best_score: the score of the best arrangement
    """
    piece_ids = list(artifacts.keys())
    best_score = float('inf')
    best_arrangement = None
    
    for perm in permutations(piece_ids):
        arrangement = list(perm)
        score = compute_puzzle_score(artifacts, arrangement, weights)
        
        if score < best_score:
            best_score = score
            best_arrangement = arrangement
    
    return best_arrangement, best_score


def swap_hillclimb(artifacts: Dict[int, dict], arrangement: list, 
                   max_iterations: int = 1000,
                   weights: SeamCostWeights = None) -> Tuple[list, float]:
    """
    Swap hillclimb refinement.
    
    Try random swaps of any two pieces. If improvement, keep it.
    """
    current = arrangement.copy()
    current_score = compute_puzzle_score(artifacts, current, weights)
    
    for _ in range(max_iterations):
        i, j = random.sample(range(4), 2)
        
        current[i], current[j] = current[j], current[i]
        new_score = compute_puzzle_score(artifacts, current, weights)
        
        if new_score < current_score:
            current_score = new_score
        else:
            current[i], current[j] = current[j], current[i]
    
    return current, current_score


def solve_2x2(artifacts: Dict[int, dict], verbose: bool = True,
              weights: SeamCostWeights = None) -> Tuple[list, float]:
    """
    Main solver function for 2x2 puzzles.
    
    REQUIRES Phase 1 artifacts - will fail without them.
    
    Args:
        artifacts: dict mapping piece_id (0-3) to artifact dict
                   Each artifact MUST contain: 'rgb', 'gray', 'edges'
        verbose: print progress info
        weights: optional custom feature weights
    
    Returns:
        arrangement: list of 4 piece ids in solved order
        score: final puzzle score
    
    Raises:
        ValueError: If artifacts are missing or invalid
    """
    # Validate artifacts (will raise if missing)
    validate_artifacts(artifacts)
    
    if verbose:
        print("=" * 50)
        print("2x2 Puzzle Solver (Artifact-First)")
        print("=" * 50)
        print(f"Pieces: {len(artifacts)}")
        print(f"Features: edges (primary), texture (primary), RGB (secondary)")
    
    # Phase 2: Exhaustive search
    if verbose:
        print("\n[Phase 2] Exhaustive search over 24 permutations...")
    
    arrangement, score = exhaustive_search(artifacts, weights)
    
    if verbose:
        print(f"  Best arrangement: {arrangement}")
        print(f"  Best score: {score:.4f}")
    
    # Phase 3: Swap hillclimb
    if verbose:
        print("\n[Phase 3] Swap hillclimb refinement...")
    
    arrangement, score = swap_hillclimb(artifacts, arrangement, 1000, weights)
    
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
