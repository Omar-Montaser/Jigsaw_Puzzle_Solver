"""
8x8 Puzzle Solver - Artifact-First Implementation

REQUIRES Phase 1 artifacts (edges, enhanced grayscale).
Uses multi-feature matching: edges (primary) + texture (primary) + RGB (secondary).

Algorithm:
- Phase 1: Build match table using multi-feature seam cost
- Phase 2: Constraint-guided beam search placement
- Phase 3: Swap hillclimb refinement
"""

import random
from typing import Dict, Tuple

from .seam_cost import build_match_table, score_board, validate_artifacts, SeamCostWeights


def beam_solve(artifacts: Dict[int, dict], match: dict,
               beam_width: int = 5000, grid_size: int = 8) -> Tuple[dict, float]:
    """
    Build the board using beam search with constraint-guided placement.
    """
    piece_ids = list(artifacts.keys())
    
    initial_state = ({}, frozenset(), 0.0)
    beam = [initial_state]
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    for pos_idx, (r, c) in enumerate(positions):
        next_beam = []
        
        for board, used, cum_score in beam:
            for pid in piece_ids:
                if pid in used:
                    continue
                
                placement_cost = 0.0
                
                if c > 0:
                    left_pid = board[(r, c - 1)]
                    placement_cost += match[left_pid][pid]['right']
                
                if r > 0:
                    top_pid = board[(r - 1, c)]
                    placement_cost += match[top_pid][pid]['bottom']
                
                new_board = board.copy()
                new_board[(r, c)] = pid
                new_used = used | {pid}
                new_score = cum_score + placement_cost
                
                next_beam.append((new_board, new_used, new_score))
        
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        if (pos_idx + 1) % grid_size == 0:
            print(f"    Row {(pos_idx + 1) // grid_size} complete: {len(next_beam)} -> {len(beam)} states")
    
    best_board, _, best_score = beam[0]
    return best_board, best_score


def swap_hillclimb(match: dict, board: dict,
                   max_iterations: int = 10000, grid_size: int = 8) -> Tuple[dict, float]:
    """Swap hillclimb refinement."""
    current = board.copy()
    current_score = score_board(match, current, grid_size)
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    improvements = 0
    for _ in range(max_iterations):
        pos1, pos2 = random.sample(positions, 2)
        
        current[pos1], current[pos2] = current[pos2], current[pos1]
        new_score = score_board(match, current, grid_size)
        
        if new_score < current_score:
            current_score = new_score
            improvements += 1
        else:
            current[pos1], current[pos2] = current[pos2], current[pos1]
    
    print(f"    Hillclimb: {improvements} improvements")
    return current, current_score


def solve_8x8(artifacts: Dict[int, dict], verbose: bool = True,
              weights: SeamCostWeights = None) -> Tuple[dict, list, float]:
    """
    Main solver function for 8x8 puzzles.
    
    REQUIRES Phase 1 artifacts - will fail without them.
    
    Args:
        artifacts: dict mapping piece_id (0-63) to artifact dict
                   Each artifact MUST contain: 'rgb', 'gray', 'edges'
        verbose: print progress info
        weights: optional custom feature weights
    
    Returns:
        board: dict mapping (row, col) -> piece_id
        arrangement: flat list of piece ids in row-major order
        score: final puzzle score
    
    Raises:
        ValueError: If artifacts are missing or invalid
    """
    validate_artifacts(artifacts)
    
    if verbose:
        print("=" * 60)
        print("8x8 Puzzle Solver (Artifact-First)")
        print("=" * 60)
        print(f"Pieces: {len(artifacts)}")
        print(f"Features: edges (primary), texture (primary), RGB (secondary)")
    
    if verbose:
        print("\n[Phase 1] Building match table from artifacts...")
    
    match = build_match_table(artifacts, weights)
    
    if verbose:
        print("  Match table computed (64x64x2 = 8192 scores)")
    
    if verbose:
        print("\n[Phase 2] Beam search placement...")
    
    board, score = beam_solve(artifacts, match, beam_width=5000, grid_size=8)
    
    if verbose:
        print(f"  Beam search score: {score:.4f}")
    
    if verbose:
        print("\n[Phase 3] Swap hillclimb refinement (10000 iterations)...")
    
    board, score = swap_hillclimb(match, board, max_iterations=10000, grid_size=8)
    
    if verbose:
        print(f"  Final score: {score:.4f}")
    
    arrangement = board_to_arrangement(board, grid_size=8)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Final arrangement: {arrangement}")
        print(f"Final score: {score:.4f}")
        print("=" * 60)
    
    return board, arrangement, score


def board_to_arrangement(board: dict, grid_size: int = 8) -> list:
    """Convert board dict to flat arrangement list."""
    return [board[(r, c)] for r in range(grid_size) for c in range(grid_size)]


def arrangement_to_board(arrangement: list, grid_size: int = 8) -> dict:
    """Convert flat arrangement to board dict."""
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board
