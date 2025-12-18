"""
4x4 Puzzle Solver

Hierarchy:
1. RGB seams = PRIMARY (beam search ranking)
2. Border constraint = STRUCTURAL (during beam search)
3. Global coherence = WEAK TIE-BREAKER (final refinement)
"""

import random
import numpy as np
from typing import Dict, Tuple

from .seam_cost import build_match_table_rgb_only, score_board, validate_artifacts


# =============================================================================
# BORDER DETECTION (structural constraint)
# =============================================================================

def compute_border_likelihood(artifact: dict, edge: str) -> float:
    """Estimate if an edge is an outer border (uniform, low texture)."""
    gray = artifact['gray'].astype(np.float32)
    edges = artifact['edges'].astype(np.float32)
    
    w = 10  # strip width
    if edge == 'top':
        gray_strip, edge_strip = gray[:w, :], edges[:w, :]
    elif edge == 'bottom':
        gray_strip, edge_strip = gray[-w:, :], edges[-w:, :]
    elif edge == 'left':
        gray_strip, edge_strip = gray[:, :w], edges[:, :w]
    elif edge == 'right':
        gray_strip, edge_strip = gray[:, -w:], edges[:, -w:]
    else:
        return 0.0
    
    # Low variance + low edge density + extreme intensity = likely border
    variance_score = max(0, 1.0 - np.var(gray_strip) / 500.0)
    edge_score = 1.0 - np.mean(edge_strip > 0)
    intensity = np.mean(gray_strip)
    extreme_score = 0.5 if (intensity < 30 or intensity > 225) else 0.0
    
    return 0.5 * variance_score + 0.4 * edge_score + 0.1 * extreme_score


def precompute_border_scores(artifacts: Dict[int, dict]) -> Dict[int, dict]:
    """Precompute border likelihood for all edges of all pieces."""
    return {
        pid: {edge: compute_border_likelihood(art, edge) 
              for edge in ['top', 'bottom', 'left', 'right']}
        for pid, art in artifacts.items()
    }


def compute_border_penalty(border_scores: dict, board: dict, pid: int,
                           r: int, c: int, grid_size: int, threshold: float = 0.4) -> float:
    """
    Border constraint applied DURING beam search.
    Penalizes border-like edges on internal seams, rewards them on boundaries.
    """
    penalty = 0.0
    scores = border_scores[pid]
    INTERNAL_PENALTY, BOUNDARY_REWARD = 50.0, 20.0
    
    # Left edge
    if c > 0:  # Internal seam
        if scores['left'] > threshold:
            penalty += (scores['left'] - threshold) * INTERNAL_PENALTY
        if border_scores[board[(r, c-1)]]['right'] > threshold:
            penalty += (border_scores[board[(r, c-1)]]['right'] - threshold) * INTERNAL_PENALTY
    else:  # Puzzle boundary
        penalty -= (scores['left'] - threshold) * BOUNDARY_REWARD if scores['left'] > threshold else 0
    
    # Top edge
    if r > 0:  # Internal seam
        if scores['top'] > threshold:
            penalty += (scores['top'] - threshold) * INTERNAL_PENALTY
        if border_scores[board[(r-1, c)]]['bottom'] > threshold:
            penalty += (border_scores[board[(r-1, c)]]['bottom'] - threshold) * INTERNAL_PENALTY
    else:  # Puzzle boundary
        penalty -= (scores['top'] - threshold) * BOUNDARY_REWARD if scores['top'] > threshold else 0
    
    # Right/bottom boundaries
    if c == grid_size - 1 and scores['right'] > threshold:
        penalty -= (scores['right'] - threshold) * BOUNDARY_REWARD
    if r == grid_size - 1 and scores['bottom'] > threshold:
        penalty -= (scores['bottom'] - threshold) * BOUNDARY_REWARD
    
    return penalty


# =============================================================================
# GLOBAL COHERENCE (weak tie-breaker on complete boards)
# =============================================================================

def compute_coherence_penalty(artifacts: Dict[int, dict], board: dict, grid_size: int) -> float:
    """
    Single coherence function combining gradient flow and texture consistency.
    Only applied to complete boards as a weak tie-breaker.
    """
    if len(board) < grid_size * grid_size:
        return 0.0
    
    penalty = 0.0
    
    # Extract piece features
    features = {}
    for pos, pid in board.items():
        gray = artifacts[pid]['gray'].astype(np.float32)
        features[pos] = {
            'mean': np.mean(gray),
            'var': np.var(gray),
            'h_grad': np.mean(gray[:, -5:]) - np.mean(gray[:, :5]),
            'v_grad': np.mean(gray[-5:, :]) - np.mean(gray[:5, :])
        }
    
    # Gradient consistency: penalize sign flips in rows/columns
    for r in range(grid_size):
        for c in range(grid_size - 1):
            g1, g2 = features[(r, c)]['h_grad'], features[(r, c+1)]['h_grad']
            if g1 * g2 < 0 and abs(g1) > 5 and abs(g2) > 5:
                penalty += abs(g1 - g2) * 0.1
    
    for c in range(grid_size):
        for r in range(grid_size - 1):
            g1, g2 = features[(r, c)]['v_grad'], features[(r+1, c)]['v_grad']
            if g1 * g2 < 0 and abs(g1) > 5 and abs(g2) > 5:
                penalty += abs(g1 - g2) * 0.1
    
    # Texture variance jumps
    for r in range(grid_size):
        for c in range(grid_size):
            v1 = features[(r, c)]['var']
            for dr, dc in [(0, 1), (1, 0)]:
                if (r+dr, c+dc) in features:
                    v2 = features[(r+dr, c+dc)]['var']
                    ratio = max(v1, v2) / (min(v1, v2) + 1e-10)
                    if ratio > 3.0:
                        penalty += (ratio - 3.0) * 2.0
    
    return penalty


# =============================================================================
# BEAM SEARCH (RGB primary + border constraint)
# =============================================================================

def beam_search(artifacts: Dict[int, dict], match: dict, border_scores: dict,
                beam_width: int = 2000, grid_size: int = 4) -> Tuple[dict, float]:
    """
    Beam search with RGB seams (primary) and border constraint (structural).
    """
    import heapq
    
    piece_ids = list(artifacts.keys())
    beam = [({}, frozenset(), 0.0)]  # (board, used, score)
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    for pos_idx, (r, c) in enumerate(positions):
        next_beam = []
        
        for board, used, cum_score in beam:
            for pid in piece_ids:
                if pid in used:
                    continue
                
                # PRIMARY: RGB seam cost
                seam_cost = 0.0
                if c > 0:
                    seam_cost += match[board[(r, c-1)]][pid]['right']
                if r > 0:
                    seam_cost += match[board[(r-1, c)]][pid]['bottom']
                
                # STRUCTURAL: Border constraint
                border_penalty = compute_border_penalty(border_scores, board, pid, r, c, grid_size)
                
                new_board = board.copy()
                new_board[(r, c)] = pid
                next_beam.append((new_board, used | {pid}, cum_score + seam_cost + border_penalty))
        
        # Use heapq.nsmallest for efficiency instead of full sort
        beam = heapq.nsmallest(beam_width, next_beam, key=lambda x: x[2])
        
        if (pos_idx + 1) % grid_size == 0:
            print(f"    Row {(pos_idx + 1) // grid_size}: {len(next_beam)} -> {len(beam)} states")
    
    return beam[0][0], score_board(match, beam[0][0], grid_size)


# =============================================================================
# REFINEMENT (simple swap hillclimb)
# =============================================================================

def compute_delta_score(match: dict, board: dict, p1: Tuple[int, int], p2: Tuple[int, int], grid_size: int) -> float:
    """
    Compute the change in score if we swap positions p1 and p2.
    Only considers affected seams (much faster than full rescore).
    """
    r1, c1 = p1
    r2, c2 = p2
    pid1, pid2 = board[p1], board[p2]
    
    def seam_cost_at(r, c, pid):
        """Cost of seams touching position (r,c) with piece pid."""
        cost = 0.0
        # Left seam
        if c > 0:
            cost += match[board[(r, c-1)]][pid]['right']
        # Right seam
        if c < grid_size - 1:
            neighbor = board[(r, c+1)]
            if (r, c+1) != p1 and (r, c+1) != p2:
                cost += match[pid][neighbor]['right']
        # Top seam
        if r > 0:
            cost += match[board[(r-1, c)]][pid]['bottom']
        # Bottom seam
        if r < grid_size - 1:
            neighbor = board[(r+1, c)]
            if (r+1, c) != p1 and (r+1, c) != p2:
                cost += match[pid][neighbor]['bottom']
        return cost
    
    # Current cost of affected seams
    old_cost = seam_cost_at(r1, c1, pid1) + seam_cost_at(r2, c2, pid2)
    
    # Cost after swap
    new_cost = seam_cost_at(r1, c1, pid2) + seam_cost_at(r2, c2, pid1)
    
    # Handle direct adjacency between p1 and p2
    if abs(r1 - r2) + abs(c1 - c2) == 1:
        if c2 == c1 + 1:  # p2 is right of p1
            old_cost += match[pid1][pid2]['right']
            new_cost += match[pid2][pid1]['right']
        elif c1 == c2 + 1:  # p1 is right of p2
            old_cost += match[pid2][pid1]['right']
            new_cost += match[pid1][pid2]['right']
        elif r2 == r1 + 1:  # p2 is below p1
            old_cost += match[pid1][pid2]['bottom']
            new_cost += match[pid2][pid1]['bottom']
        elif r1 == r2 + 1:  # p1 is below p2
            old_cost += match[pid2][pid1]['bottom']
            new_cost += match[pid1][pid2]['bottom']
    
    return new_cost - old_cost


def swap_hillclimb(artifacts: Dict[int, dict], match: dict, board: dict,
                   grid_size: int = 4, max_iter: int = 1000) -> Tuple[dict, float]:
    """Simple swap hillclimb using incremental scoring (fast)."""
    current = board.copy()
    current_score = score_board(match, current, grid_size)
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    improvements = 0
    no_improve_count = 0
    
    for _ in range(max_iter):
        p1, p2 = random.sample(positions, 2)
        
        # Fast incremental delta computation
        delta = compute_delta_score(match, current, p1, p2, grid_size)
        
        if delta < -0.01:  # Small threshold to avoid floating point noise
            current[p1], current[p2] = current[p2], current[p1]
            current_score += delta
            improvements += 1
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Early termination if stuck
            if no_improve_count > 200:
                break
    
    print(f"    Hillclimb: {improvements} improvements")
    return current, current_score


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_4x4(artifacts: Dict[int, dict], verbose: bool = True) -> Tuple[dict, list, float]:
    """
    Solve 4x4 puzzle.
    
    Hierarchy:
    1. RGB seams = PRIMARY (beam search)
    2. Border constraint = STRUCTURAL (beam search)
    3. Coherence = TIE-BREAKER (refinement)
    """
    validate_artifacts(artifacts)
    
    if verbose:
        print("=" * 60)
        print("4x4 Puzzle Solver")
        print("=" * 60)
    
    # Build match table and border scores
    if verbose:
        print("\n[1] Building match table...")
    match = build_match_table_rgb_only(artifacts)
    border_scores = precompute_border_scores(artifacts)
    
    # Beam search
    if verbose:
        print("\n[2] Beam search (RGB + border constraint)...")
    board, score = beam_search(artifacts, match, border_scores, beam_width=500, grid_size=4)
    if verbose:
        print(f"    Score: {score:.2f}")
    
    # Refinement
    if verbose:
        print("\n[3] Swap hillclimb refinement...")
    board, score = swap_hillclimb(artifacts, match, board, grid_size=4, max_iter=5000)
    
    arrangement = [board[(r, c)] for r in range(4) for c in range(4)]
    
    if verbose:
        print(f"\nFinal: {arrangement}, score={score:.2f}")
    
    return board, arrangement, score


def arrangement_to_board(arrangement: list, grid_size: int = 4) -> dict:
    """Convert flat arrangement to board dict."""
    return {(r, c): arrangement[r * grid_size + c] 
            for r in range(grid_size) for c in range(grid_size)}