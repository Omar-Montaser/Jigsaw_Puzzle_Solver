"""
Structural Sanity Evaluation for 8x8 Puzzle Solver

Evaluates solution quality using LOCAL, RELATIVE, SOFT-SCORED metrics.
No ground-truth comparison - purely based on edge compatibility scores.

Metrics:
1. Local rank score: How well each edge ranks among possible neighbors
2. Symmetric rank agreement: Soft mutual-best scoring
3. Adaptive coherent regions: Connected components using relative thresholds

Expected scores:
- Random shuffle: 0-10%
- Partial coherence: 30-70%
- Near-perfect: 90-100%
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import deque
from dataclasses import dataclass


@dataclass
class SanityMetrics:
    """Container for all sanity metrics."""
    local_rank_score: float       # [0, 1] - average rank-based edge quality
    symmetric_agreement: float    # [0, 1] - soft mutual ranking agreement
    coherent_region: float        # [0, 1] - largest coherent region / total
    final_score: float            # [0, 100] - weighted combination
    
    total_edges: int
    avg_rank_percentile: float
    region_size: int


def compute_local_rank_score(
    board: Dict[Tuple[int, int], int],
    compat,  # CompatibilityMatrix
    grid_size: int
) -> Tuple[float, float, int]:
    """
    Compute local rank-based edge quality.
    
    For each placed edge (A, B):
    - Compute B's rank among all possible right/bottom neighbors of A
    - Score = (1 - rank_pct)^2  (squared to penalize bad ranks more)
    - Average across all edges
    
    This makes "better than random" edges score positively,
    while heavily penalizing poor matches.
    
    Returns:
        (score, avg_rank_percentile, total_edges)
    """
    n_pieces = grid_size * grid_size
    
    # Precompute sorted neighbor rankings for each piece
    h_ranks = {}  # pid -> {neighbor: rank} for horizontal (right)
    v_ranks = {}  # pid -> {neighbor: rank} for vertical (bottom)
    
    for pid in range(n_pieces):
        h_scores = [(j, compat.get_horizontal_score(pid, j)) for j in range(n_pieces) if j != pid]
        h_scores.sort(key=lambda x: x[1])
        h_ranks[pid] = {neighbor: rank for rank, (neighbor, _) in enumerate(h_scores)}
        
        v_scores = [(j, compat.get_vertical_score(pid, j)) for j in range(n_pieces) if j != pid]
        v_scores.sort(key=lambda x: x[1])
        v_ranks[pid] = {neighbor: rank for rank, (neighbor, _) in enumerate(v_scores)}
    
    scores = []
    rank_percentiles = []
    
    for (row, col), pid in board.items():
        # Right neighbor (horizontal edge)
        if (row, col + 1) in board:
            right_pid = board[(row, col + 1)]
            rank = h_ranks[pid].get(right_pid, n_pieces - 1)
            rank_pct = rank / (n_pieces - 1)
            # Tiered scoring: top 5% = 1.0, top 15% = 0.5, top 30% = 0.15, else = 0
            if rank_pct < 0.05:
                score = 1.0
            elif rank_pct < 0.15:
                score = 0.5
            elif rank_pct < 0.30:
                score = 0.15
            else:
                score = 0.0
            scores.append(score)
            rank_percentiles.append(rank_pct)
        
        # Bottom neighbor (vertical edge)
        if (row + 1, col) in board:
            bottom_pid = board[(row + 1, col)]
            rank = v_ranks[pid].get(bottom_pid, n_pieces - 1)
            rank_pct = rank / (n_pieces - 1)
            if rank_pct < 0.05:
                score = 1.0
            elif rank_pct < 0.15:
                score = 0.5
            elif rank_pct < 0.30:
                score = 0.15
            else:
                score = 0.0
            scores.append(score)
            rank_percentiles.append(rank_pct)
    
    avg_score = np.mean(scores) if scores else 0.0
    avg_rank_pct = np.mean(rank_percentiles) if rank_percentiles else 1.0
    
    return avg_score, avg_rank_pct, len(scores)


def compute_symmetric_agreement(
    board: Dict[Tuple[int, int], int],
    compat,  # CompatibilityMatrix
    grid_size: int,
    top_threshold: float = 0.20  # Top 20%
) -> float:
    """
    Compute soft symmetric rank agreement.
    
    For each edge (A, B):
    - Compute A's rank of B and B's rank of A
    - If both in top 20%: score = 1.0
    - If one in top 20%: score = 0.5
    - Otherwise: score = 0.25 * (1 - avg_rank_pct)
    
    This gives partial credit instead of binary pass/fail.
    
    Returns:
        Average agreement score [0, 1]
    """
    n_pieces = grid_size * grid_size
    top_k = int(n_pieces * top_threshold)
    
    # Precompute rankings
    h_right_ranks = {}  # pid -> {neighbor: rank} for pid's right edge
    h_left_ranks = {}   # pid -> {neighbor: rank} for pid's left edge
    v_bottom_ranks = {} # pid -> {neighbor: rank} for pid's bottom edge
    v_top_ranks = {}    # pid -> {neighbor: rank} for pid's top edge
    
    for pid in range(n_pieces):
        # Right edge: who fits best to my right?
        scores = [(j, compat.get_horizontal_score(pid, j)) for j in range(n_pieces) if j != pid]
        scores.sort(key=lambda x: x[1])
        h_right_ranks[pid] = {n: r for r, (n, _) in enumerate(scores)}
        
        # Left edge: who fits best to my left?
        scores = [(j, compat.get_horizontal_score(j, pid)) for j in range(n_pieces) if j != pid]
        scores.sort(key=lambda x: x[1])
        h_left_ranks[pid] = {n: r for r, (n, _) in enumerate(scores)}
        
        # Bottom edge: who fits best below me?
        scores = [(j, compat.get_vertical_score(pid, j)) for j in range(n_pieces) if j != pid]
        scores.sort(key=lambda x: x[1])
        v_bottom_ranks[pid] = {n: r for r, (n, _) in enumerate(scores)}
        
        # Top edge: who fits best above me?
        scores = [(j, compat.get_vertical_score(j, pid)) for j in range(n_pieces) if j != pid]
        scores.sort(key=lambda x: x[1])
        v_top_ranks[pid] = {n: r for r, (n, _) in enumerate(scores)}
    
    scores = []
    
    for (row, col), pid in board.items():
        # Right neighbor
        if (row, col + 1) in board:
            right_pid = board[(row, col + 1)]
            
            # A ranks B for right, B ranks A for left
            rank_a = h_right_ranks[pid].get(right_pid, n_pieces - 1)
            rank_b = h_left_ranks[right_pid].get(pid, n_pieces - 1)
            
            a_in_top = rank_a < top_k
            b_in_top = rank_b < top_k
            
            if a_in_top and b_in_top:
                score = 1.0
            elif a_in_top or b_in_top:
                score = 0.4
            else:
                # Minimal partial credit for poor matches
                avg_rank_pct = (rank_a + rank_b) / (2 * (n_pieces - 1))
                score = 0.1 * (1.0 - avg_rank_pct)
            
            scores.append(score)
        
        # Bottom neighbor
        if (row + 1, col) in board:
            bottom_pid = board[(row + 1, col)]
            
            rank_a = v_bottom_ranks[pid].get(bottom_pid, n_pieces - 1)
            rank_b = v_top_ranks[bottom_pid].get(pid, n_pieces - 1)
            
            a_in_top = rank_a < top_k
            b_in_top = rank_b < top_k
            
            if a_in_top and b_in_top:
                score = 1.0
            elif a_in_top or b_in_top:
                score = 0.4
            else:
                avg_rank_pct = (rank_a + rank_b) / (2 * (n_pieces - 1))
                score = 0.1 * (1.0 - avg_rank_pct)
            
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def compute_coherent_region(
    board: Dict[Tuple[int, int], int],
    compat,  # CompatibilityMatrix
    grid_size: int
) -> Tuple[float, int]:
    """
    Compute largest coherent connected component using GLOBAL threshold.
    
    Threshold = 15th percentile of ALL possible edge costs.
    Only truly good edges (top 15%) form connections.
    
    Returns:
        (normalized_size, component_size)
    """
    n_pieces = grid_size * grid_size
    
    # Compute global threshold from ALL possible edges
    all_h_costs = []
    all_v_costs = []
    for i in range(n_pieces):
        for j in range(n_pieces):
            if i != j:
                all_h_costs.append(compat.get_horizontal_score(i, j))
                all_v_costs.append(compat.get_vertical_score(i, j))
    
    # Use 15th percentile of all costs as threshold (only top 15% edges connect)
    h_threshold = np.percentile(all_h_costs, 15)
    v_threshold = np.percentile(all_v_costs, 15)
    
    # Collect edges in the solution
    edge_positions = []
    
    for (row, col), pid in board.items():
        # Right neighbor
        if (row, col + 1) in board:
            right_pid = board[(row, col + 1)]
            cost = compat.get_horizontal_score(pid, right_pid)
            is_good = cost <= h_threshold
            edge_positions.append(((row, col), (row, col + 1), is_good))
        
        # Bottom neighbor
        if (row + 1, col) in board:
            bottom_pid = board[(row + 1, col)]
            cost = compat.get_vertical_score(pid, bottom_pid)
            is_good = cost <= v_threshold
            edge_positions.append(((row, col), (row + 1, col), is_good))
    
    if not edge_positions:
        return 0.0, 0
    
    # Build adjacency list for good edges only
    adj_list: Dict[Tuple[int, int], List[Tuple[int, int]]] = {pos: [] for pos in board.keys()}
    
    for pos1, pos2, is_good in edge_positions:
        if is_good:
            adj_list[pos1].append(pos2)
            adj_list[pos2].append(pos1)
    
    # Find largest connected component using BFS
    visited = set()
    largest_size = 0
    
    for start_pos in board.keys():
        if start_pos in visited:
            continue
        
        queue = deque([start_pos])
        component_size = 0
        
        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            component_size += 1
            
            for neighbor in adj_list[pos]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        largest_size = max(largest_size, component_size)
    
    normalized = largest_size / n_pieces
    return normalized, largest_size


def evaluate_sanity(
    board: Dict[Tuple[int, int], int],
    compat,  # CompatibilityMatrix
    grid_size: int = 8,
    verbose: bool = False
) -> SanityMetrics:
    """
    Compute comprehensive sanity score for puzzle solution.
    
    Uses local, relative, soft-scored metrics that:
    - Give partial credit for partially correct solutions
    - Don't require ground-truth comparison
    - Adapt to each image's difficulty
    
    Final score = 0.4 * local_rank + 0.35 * symmetric + 0.25 * region
    
    Expected ranges:
    - Random: 0-10%
    - Partial: 30-70%
    - Near-perfect: 90-100%
    """
    # Metric 1: Local rank-based edge quality
    local_rank, avg_rank_pct, total_edges = compute_local_rank_score(board, compat, grid_size)
    
    # Metric 2: Symmetric rank agreement
    symmetric = compute_symmetric_agreement(board, compat, grid_size)
    
    # Metric 3: Adaptive coherent region
    region, region_size = compute_coherent_region(board, compat, grid_size)
    
    # Weighted final score
    # Adjust weights to penalize random more: region has more weight
    raw_score = 0.35 * local_rank + 0.30 * symmetric + 0.35 * region
    
    # Scale to 0-100
    final_score = raw_score * 100
    
    metrics = SanityMetrics(
        local_rank_score=local_rank,
        symmetric_agreement=symmetric,
        coherent_region=region,
        final_score=final_score,
        total_edges=total_edges,
        avg_rank_percentile=avg_rank_pct,
        region_size=region_size
    )
    
    if verbose:
        print(f"  Local rank score: {local_rank*100:.1f}% (avg rank pct: {avg_rank_pct*100:.1f}%)")
        print(f"  Symmetric agreement: {symmetric*100:.1f}%")
        print(f"  Coherent region: {region*100:.1f}% ({region_size}/{grid_size*grid_size} pieces)")
        print(f"  Final sanity score: {final_score:.1f}%")
    
    return metrics


# =====================================================
# SANITY TESTS
# =====================================================

def run_sanity_tests():
    """Test the metrics with synthetic cases."""
    print("=" * 60)
    print("SANITY TESTS (Synthetic)")
    print("=" * 60)
    
    # Create a mock CompatibilityMatrix for testing
    class MockCompat:
        def __init__(self, n_pieces, seed=42):
            np.random.seed(seed)
            self.n = n_pieces
            # Random costs, but make adjacent pieces (in ground truth) have lower costs
            self.h_costs = np.random.rand(n_pieces, n_pieces) * 0.5 + 0.25
            self.v_costs = np.random.rand(n_pieces, n_pieces) * 0.5 + 0.25
            
            # Make ground-truth neighbors have very low costs
            gs = int(np.sqrt(n_pieces))
            for pid in range(n_pieces):
                row, col = pid // gs, pid % gs
                if col < gs - 1:
                    right = pid + 1
                    self.h_costs[pid, right] = np.random.rand() * 0.05
                if row < gs - 1:
                    bottom = pid + gs
                    self.v_costs[pid, bottom] = np.random.rand() * 0.05
        
        def get_horizontal_score(self, a, b):
            return self.h_costs[a, b]
        
        def get_vertical_score(self, a, b):
            return self.v_costs[a, b]
    
    gs = 8
    n = gs * gs
    compat = MockCompat(n)
    
    # Test 1: Perfect solution
    print("\nTest 1: Perfect solution")
    perfect_board = {(r, c): r * gs + c for r in range(gs) for c in range(gs)}
    metrics = evaluate_sanity(perfect_board, compat, gs, verbose=True)
    print(f"  Expected: >90%")
    
    # Test 2: Cyclic shift by 1 row
    print("\nTest 2: Cyclic shift by 1 row")
    shifted_board = {}
    for r in range(gs):
        for c in range(gs):
            orig_r = (r - 1) % gs
            shifted_board[(r, c)] = orig_r * gs + c
    metrics = evaluate_sanity(shifted_board, compat, gs, verbose=True)
    print(f"  Expected: ~80-95% (most neighbors preserved)")
    
    # Test 3: Random shuffle
    print("\nTest 3: Random shuffle")
    import random
    random.seed(123)
    pids = list(range(n))
    random.shuffle(pids)
    random_board = {(r, c): pids[r * gs + c] for r in range(gs) for c in range(gs)}
    metrics = evaluate_sanity(random_board, compat, gs, verbose=True)
    print(f"  Expected: <15%")
    
    # Test 4: Half correct
    print("\nTest 4: Half correct (top 4 rows perfect, bottom 4 shuffled)")
    half_board = {}
    bottom_pids = list(range(32, 64))
    random.shuffle(bottom_pids)
    for r in range(gs):
        for c in range(gs):
            if r < 4:
                half_board[(r, c)] = r * gs + c
            else:
                half_board[(r, c)] = bottom_pids[(r - 4) * gs + c]
    metrics = evaluate_sanity(half_board, compat, gs, verbose=True)
    print(f"  Expected: 40-60%")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    # Run sanity tests first
    run_sanity_tests()
    
    print("\n" + "=" * 60)
    print("TESTING WITH ACTUAL SOLVER")
    print("=" * 60)
    
    from pathlib import Path
    from improvements_8x8 import solve_puzzle, SolverConfig, CompatibilityMatrix
    import cv2
    
    PUZZLE_FOLDER = "./Gravity Falls/puzzle_8x8"
    test_ids = [4, 6, 8, 14, 15, 16, 17, 18, 19]
    
    print(f"\nTesting images: {test_ids}")
    print(f"{'ID':<6} | {'Rank%':<8} | {'Symm%':<8} | {'Region%':<9} | {'Sanity':<8}")
    print("-" * 55)
    
    config = SolverConfig(verbose=False)
    results = []
    
    for img_id in test_ids:
        puzzle_path = Path(PUZZLE_FOLDER) / f"{img_id}.jpg"
        
        if not puzzle_path.exists():
            print(f"{img_id:<6} | NOT FOUND")
            continue
        
        print(f"Solving {img_id}...", end=' ', flush=True)
        
        try:
            result = solve_puzzle(str(puzzle_path), config)
            board = result['board']
            
            # Recreate CompatibilityMatrix for evaluation
            img = cv2.imread(str(puzzle_path))
            h, w = img.shape[:2]
            gs = 8
            ph, pw = h // gs, w // gs
            
            pieces = {}
            for idx in range(gs * gs):
                r, c = idx // gs, idx % gs
                pieces[idx] = img[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()
            
            compat = CompatibilityMatrix(pieces, config)
            metrics = evaluate_sanity(board, compat, gs)
            
            results.append({'id': img_id, 'metrics': metrics})
            
            print(f"\r{img_id:<6} | {metrics.local_rank_score*100:>6.1f}% | {metrics.symmetric_agreement*100:>6.1f}% | {metrics.coherent_region*100:>7.1f}% | {metrics.final_score:>6.1f}%")
            
        except Exception as e:
            print(f"\r{img_id:<6} | ERROR: {e}")
    
    if results:
        avg_sanity = np.mean([r['metrics'].final_score for r in results])
        avg_rank = np.mean([r['metrics'].local_rank_score for r in results])
        avg_symm = np.mean([r['metrics'].symmetric_agreement for r in results])
        avg_region = np.mean([r['metrics'].coherent_region for r in results])
        
        print("-" * 55)
        print(f"{'AVG':<6} | {avg_rank*100:>6.1f}% | {avg_symm*100:>6.1f}% | {avg_region*100:>7.1f}% | {avg_sanity:>6.1f}%")
