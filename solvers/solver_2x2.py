"""
2x2 Puzzle Solver - RGB-Only Implementation

Simple RGB-based matching for 2x2 puzzles.
Uses SSD + NCC + continuity scoring on raw RGB pixels.

Algorithm:
- Exhaustive search over all 24 permutations
- Border constraint to break ties on uniform backgrounds
- Swap hillclimb refinement (rarely changes anything)
"""

import random
import numpy as np
from itertools import permutations
from typing import Dict, Tuple


# =============================================================================
# BORDER DETECTION (structural constraint for uniform backgrounds)
# =============================================================================

def compute_border_likelihood(piece: np.ndarray, edge: str) -> float:
    """
    Estimate if an edge is an outer border (uniform, low texture).
    RGB-based only - no Canny or other feature pipelines.
    """
    gray = np.mean(piece, axis=2).astype(np.float32)
    w = 10  # strip width
    
    if edge == 'top':
        strip = gray[:w, :]
    elif edge == 'bottom':
        strip = gray[-w:, :]
    elif edge == 'left':
        strip = gray[:, :w]
    elif edge == 'right':
        strip = gray[:, -w:]
    else:
        return 0.0
    
    # Low variance = likely border
    variance_score = max(0, 1.0 - np.var(strip) / 500.0)
    
    # Extreme intensity (very dark or very bright) = weak border signal
    intensity = np.mean(strip)
    extreme_score = 0.3 if (intensity < 30 or intensity > 225) else 0.0
    
    return 0.7 * variance_score + 0.3 * extreme_score


def precompute_border_scores(pieces: Dict[int, np.ndarray]) -> Dict[int, dict]:
    """Precompute border likelihood for all edges of all pieces."""
    return {
        pid: {edge: compute_border_likelihood(piece, edge) 
              for edge in ['top', 'bottom', 'left', 'right']}
        for pid, piece in pieces.items()
    }


def compute_border_penalty(border_scores: dict, arrangement: list, threshold: float = 0.3) -> float:
    """
    Border constraint for 2x2 puzzles.
    Strengthened for 2x2 to break symmetric tie cases on uniform backgrounds.
    
    Layout:
        P[0]  P[1]
        P[2]  P[3]
    
    Internal seams: P[0]-right/P[1]-left, P[2]-right/P[3]-left (horizontal)
                    P[0]-bottom/P[2]-top, P[1]-bottom/P[3]-top (vertical)
    
    Outer boundaries: P[0]-top, P[0]-left, P[1]-top, P[1]-right,
                      P[2]-bottom, P[2]-left, P[3]-bottom, P[3]-right
    """
    # TASK 1: Strengthened constants for 2x2 to break symmetry ties
    INTERNAL_PENALTY = 80.0   # Increased from 50.0 - stronger penalty for border on internal seams
    BOUNDARY_REWARD = 35.0    # Increased from 20.0 - stronger reward for border on boundaries
    penalty = 0.0
    
    P = arrangement
    
    # Internal seams - penalize border-like edges
    internal_edges = [
        (P[0], 'right'), (P[1], 'left'),    # horizontal seam top
        (P[2], 'right'), (P[3], 'left'),    # horizontal seam bottom
        (P[0], 'bottom'), (P[2], 'top'),    # vertical seam left
        (P[1], 'bottom'), (P[3], 'top'),    # vertical seam right
    ]
    for pid, edge in internal_edges:
        score = border_scores[pid][edge]
        if score > threshold:
            penalty += (score - threshold) * INTERNAL_PENALTY
    
    # Outer boundaries - reward border-like edges
    boundary_edges = [
        (P[0], 'top'), (P[0], 'left'),
        (P[1], 'top'), (P[1], 'right'),
        (P[2], 'bottom'), (P[2], 'left'),
        (P[3], 'bottom'), (P[3], 'right'),
    ]
    for pid, edge in boundary_edges:
        score = border_scores[pid][edge]
        if score > threshold:
            penalty -= (score - threshold) * BOUNDARY_REWARD
    
    return penalty


# =============================================================================
# EDGE EXTRACTION AND SEAM COST
# =============================================================================

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
    
    # TASK 3: Variance-normalized SSD to prevent flat backgrounds from dominating
    # Flat edges have low variance, so their SSD is artificially low - normalize by variance
    var_a = np.var(edgeA_pixels) + 1e-10
    var_b = np.var(edgeB_pixels) + 1e-10
    edge_variance = var_a + var_b
    
    # 1. SSD - Sum of Squared Differences, normalized by edge variance
    raw_ssd = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    ssd_score = raw_ssd / (edge_variance / 1000.0 + 1.0)  # Normalize: flat edges get boosted SSD
    
    # 2. Normalized Cross-Correlation
    mean_a = np.mean(edgeA_pixels)
    mean_b = np.mean(edgeB_pixels)
    std_a = np.std(edgeA_pixels) + 1e-10
    std_b = np.std(edgeB_pixels) + 1e-10
    ncc = np.mean((edgeA_pixels - mean_a) * (edgeB_pixels - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100
    
    # 3. Gradient continuity
    var_near_a = np.var(stripA_near.flatten()) + 1e-10
    var_near_b = np.var(stripB_near.flatten()) + 1e-10
    seam_diff = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    continuity_score = seam_diff / ((var_near_a + var_near_b) / 2)
    
    return 0.3 * np.sqrt(ssd_score) + 0.3 * ncc_cost + 0.4 * continuity_score * 10


def compute_seam_score(pieces: Dict[int, np.ndarray], arrangement: list) -> float:
    """
    Compute seam cost only for a 2x2 arrangement.
    
    Arrangement layout:
        P[0]  P[1]
        P[2]  P[3]
    """
    P = arrangement
    
    # Individual seam costs
    h_top = seam_cost(pieces, P[0], 'right', P[1], 'left')     # horizontal seam top row
    h_bot = seam_cost(pieces, P[2], 'right', P[3], 'left')     # horizontal seam bottom row
    v_left = seam_cost(pieces, P[0], 'bottom', P[2], 'top')    # vertical seam left col
    v_right = seam_cost(pieces, P[1], 'bottom', P[3], 'top')   # vertical seam right col
    
    # Base seam score
    score = h_top + h_bot + v_left + v_right
    
    # TASK 2: Seam orientation consistency penalty
    # In a correct solution, paired seams should have similar costs
    # This breaks ties when symmetric arrangements have equal total seam cost
    CONSISTENCY_WEIGHT = 0.25  # Small factor - only a tie-breaker
    h_diff = abs(h_top - h_bot)  # Horizontal seams should be similar
    v_diff = abs(v_left - v_right)  # Vertical seams should be similar
    consistency_penalty = (h_diff + v_diff) * CONSISTENCY_WEIGHT
    
    return score + consistency_penalty


def compute_puzzle_score(pieces: Dict[int, np.ndarray], arrangement: list,
                         border_scores: dict = None) -> float:
    """
    Compute total puzzle score = seam_score + border_penalty.
    
    Seam cost is PRIMARY, border penalty only breaks ties.
    """
    seam_score = compute_seam_score(pieces, arrangement)
    
    if border_scores is not None:
        border_penalty = compute_border_penalty(border_scores, arrangement)
        return seam_score + border_penalty
    
    return seam_score


def exhaustive_search(pieces: Dict[int, np.ndarray], border_scores: dict) -> Tuple[list, float]:
    """Try all 24 permutations and find the one with lowest score."""
    piece_ids = list(pieces.keys())
    best_score = float('inf')
    best_arrangement = None
    
    for perm in permutations(piece_ids):
        arrangement = list(perm)
        score = compute_puzzle_score(pieces, arrangement, border_scores)
        if score < best_score:
            best_score = score
            best_arrangement = arrangement
    
    return best_arrangement, best_score


def swap_hillclimb(pieces: Dict[int, np.ndarray], arrangement: list,
                   border_scores: dict, max_iterations: int = 1000) -> Tuple[list, float]:
    """Swap hillclimb refinement."""
    current = arrangement.copy()
    current_score = compute_puzzle_score(pieces, current, border_scores)
    
    for _ in range(max_iterations):
        i, j = random.sample(range(4), 2)
        current[i], current[j] = current[j], current[i]
        new_score = compute_puzzle_score(pieces, current, border_scores)
        
        if new_score < current_score:
            current_score = new_score
        else:
            current[i], current[j] = current[j], current[i]
    
    return current, current_score


def solve_2x2(artifacts: Dict[int, dict], verbose: bool = True) -> Tuple[list, float]:
    """
    Main solver function for 2x2 puzzles.
    
    Uses RGB from artifacts for simple SSD+NCC+continuity matching.
    Border constraint breaks ties on uniform backgrounds.
    
    Args:
        artifacts: dict mapping piece_id (0-3) to artifact dict
                   Each artifact MUST contain: 'rgb'
        verbose: print progress info
    
    Returns:
        arrangement: list of 4 piece ids in solved order
        score: final puzzle score (seam cost only, for comparison)
    """
    # Extract RGB pieces from artifacts
    pieces = {pid: art['rgb'] for pid, art in artifacts.items()}
    
    if verbose:
        print("=" * 50)
        print("2x2 Puzzle Solver (RGB + Border Constraint)")
        print("=" * 50)
        print(f"Pieces: {len(pieces)}")
    
    # Precompute border scores
    border_scores = precompute_border_scores(pieces)
    
    # Exhaustive search with border constraint
    if verbose:
        print("\nExhaustive search over 24 permutations...")
    
    arrangement, score = exhaustive_search(pieces, border_scores)
    
    if verbose:
        print(f"  Best arrangement: {arrangement}")
        print(f"  Best score: {score:.4f}")
    
    # Swap hillclimb with border constraint
    if verbose:
        print("\nSwap hillclimb refinement...")
    
    arrangement, score = swap_hillclimb(pieces, arrangement, border_scores, 1000)
    
    if verbose:
        print(f"  Final arrangement: {arrangement}")
        print(f"  Final score: {score:.4f}")
    
    # Return seam score only (without border penalty) for external comparison
    final_seam_score = compute_seam_score(pieces, arrangement)
    
    return arrangement, final_seam_score


def arrangement_to_board(arrangement: list, grid_size: int = 2) -> dict:
    """Convert flat arrangement to board dict format."""
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board
