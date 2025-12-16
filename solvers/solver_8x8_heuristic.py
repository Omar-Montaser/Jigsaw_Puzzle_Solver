"""
Lightweight 8x8 Puzzle Solver - Row-Based Assembly with Border Constraints

Architecture:
1. Precompute edge costs (horizontal, vertical) + border scores
2. Build candidate ROWS using beam search (left→right)
3. Select 8 disjoint rows greedily
4. Stack rows optimally using vertical compatibility
5. Refine with random swap hill-climb

This approach enforces global structure early by building complete rows
before vertical assembly, avoiding the pitfall of greedy cell-by-cell placement.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from itertools import permutations

GRID_SIZE = 8
N_PIECES = 64
STRIP_WIDTH = 2

# Search parameters
ROW_BEAM_WIDTH = 3000
MAX_REFINE_SWAPS = 50000
TIME_LIMIT = 30.0

# Border constraint weight (low - structural guidance only)
BORDER_WEIGHT = 0.01


def extract_edge(piece: np.ndarray, edge: str) -> np.ndarray:
    """Extract pixel strip from piece edge."""
    if edge == 'top':
        return piece[:STRIP_WIDTH, :].flatten().astype(np.float32)
    elif edge == 'bottom':
        return piece[-STRIP_WIDTH:, :].flatten().astype(np.float32)
    elif edge == 'left':
        return piece[:, :STRIP_WIDTH].flatten().astype(np.float32)
    elif edge == 'right':
        return piece[:, -STRIP_WIDTH:].flatten().astype(np.float32)
    return np.array([])


def compute_border_score(piece: np.ndarray, edge: str) -> float:
    """
    Compute border likelihood for an edge.
    True puzzle borders have LOW gradient variance (smooth/uniform).
    Returns normalized score: LOWER = more likely to be true border.
    """
    if edge == 'top':
        strip = piece[:3, :]
    elif edge == 'bottom':
        strip = piece[-3:, :]
    elif edge == 'left':
        strip = piece[:, :3]
    elif edge == 'right':
        strip = piece[:, -3:]
    else:
        return 1.0
    
    if len(strip.shape) == 3:
        gray = np.mean(strip, axis=2)
    else:
        gray = strip.astype(np.float32)
    
    # Gradient variance - low = likely border
    gx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    
    var = np.var(gx) + np.var(gy)
    return var


def compute_edge_costs(pieces: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Precompute edge compatibility and border scores.
    Returns: (h_costs, v_costs, border_scores)
    """
    n = len(pieces)
    h_costs = np.full((n, n), np.inf, dtype=np.float32)
    v_costs = np.full((n, n), np.inf, dtype=np.float32)
    
    # Extract all edges
    right_edges = {i: extract_edge(pieces[i], 'right') for i in range(n)}
    left_edges = {i: extract_edge(pieces[i], 'left') for i in range(n)}
    bottom_edges = {i: extract_edge(pieces[i], 'bottom') for i in range(n)}
    top_edges = {i: extract_edge(pieces[i], 'top') for i in range(n)}
    
    # Compute border scores for each piece edge
    border_scores = {}
    for i in range(n):
        for edge in ['top', 'bottom', 'left', 'right']:
            border_scores[(i, edge)] = compute_border_score(pieces[i], edge)
    
    # Normalize border scores to [0, 1]
    all_scores = list(border_scores.values())
    min_s, max_s = min(all_scores), max(all_scores)
    range_s = max_s - min_s if max_s > min_s else 1.0
    for k in border_scores:
        border_scores[k] = (border_scores[k] - min_s) / range_s
    
    # Compute pairwise costs
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            h_costs[a, b] = np.mean(np.abs(right_edges[a] - left_edges[b])) / 255.0
            v_costs[a, b] = np.mean(np.abs(bottom_edges[a] - top_edges[b])) / 255.0
    
    return h_costs, v_costs, border_scores


def get_border_penalty(pid: int, row: int, col: int, border_scores: Dict) -> float:
    """
    Compute border penalty for placing piece at position.
    Penalize non-border edges on puzzle boundaries.
    Reward border-like edges on puzzle boundaries.
    """
    penalty = 0.0
    gs = GRID_SIZE
    
    # Top boundary
    if row == 0:
        # Reward low border score (likely true border)
        penalty -= BORDER_WEIGHT * (1.0 - border_scores[(pid, 'top')])
    else:
        # Penalize border-like edge on internal position
        penalty += BORDER_WEIGHT * (1.0 - border_scores[(pid, 'top')]) * 0.3
    
    # Bottom boundary
    if row == gs - 1:
        penalty -= BORDER_WEIGHT * (1.0 - border_scores[(pid, 'bottom')])
    else:
        penalty += BORDER_WEIGHT * (1.0 - border_scores[(pid, 'bottom')]) * 0.3
    
    # Left boundary
    if col == 0:
        penalty -= BORDER_WEIGHT * (1.0 - border_scores[(pid, 'left')])
    else:
        penalty += BORDER_WEIGHT * (1.0 - border_scores[(pid, 'left')]) * 0.3
    
    # Right boundary
    if col == gs - 1:
        penalty -= BORDER_WEIGHT * (1.0 - border_scores[(pid, 'right')])
    else:
        penalty += BORDER_WEIGHT * (1.0 - border_scores[(pid, 'right')]) * 0.3
    
    return penalty


def build_candidate_rows(h_costs: np.ndarray, border_scores: Dict) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Build candidate rows using beam search (left→right).
    Each row is a complete sequence of 8 pieces.
    Returns list of (row_tuple, avg_cost).
    """
    gs = GRID_SIZE
    piece_ids = list(range(N_PIECES))
    
    # State: (row_tuple, used_set, cumulative_cost)
    # Start with each piece as potential row start
    states = []
    for pid in piece_ids:
        # Add border penalty for left edge (col=0)
        border_pen = get_border_penalty(pid, 0, 0, border_scores)  # row unknown, use 0
        states.append((tuple([pid]), frozenset([pid]), border_pen))
    
    # Extend rows left→right
    for col in range(1, gs):
        new_states = []
        for row, used, cost in states:
            last_pid = row[-1]
            for pid in piece_ids:
                if pid in used:
                    continue
                edge_cost = h_costs[last_pid, pid]
                # Add border penalty for right edge if last column
                border_pen = 0.0
                if col == gs - 1:
                    border_pen = get_border_penalty(pid, 0, col, border_scores)
                new_cost = cost + edge_cost + border_pen
                new_states.append((row + (pid,), used | {pid}, new_cost))
        
        # Keep top candidates
        new_states.sort(key=lambda x: x[2] / col)
        states = new_states[:ROW_BEAM_WIDTH]
    
    # Convert to (row, avg_cost) format
    rows = [(row, cost / (gs - 1)) for row, used, cost in states]
    rows.sort(key=lambda x: x[1])
    
    return rows


def select_disjoint_rows(rows: List[Tuple[Tuple[int, ...], float]]) -> Optional[List[Tuple[int, ...]]]:
    """
    Greedily select 8 non-overlapping rows.
    Returns None if not possible.
    """
    selected = []
    used_pieces = set()
    
    for row, cost in rows:
        row_pieces = set(row)
        if row_pieces & used_pieces:
            continue
        selected.append(row)
        used_pieces |= row_pieces
        if len(selected) == GRID_SIZE:
            break
    
    return selected if len(selected) == GRID_SIZE else None


def stack_rows_optimally(rows: List[Tuple[int, ...]], v_costs: np.ndarray, 
                         border_scores: Dict) -> Dict[Tuple[int, int], int]:
    """
    Find optimal vertical stacking order for rows.
    Uses exhaustive search over 8! permutations (40320 - feasible).
    """
    gs = GRID_SIZE
    
    def row_vertical_cost(row_top: Tuple[int, ...], row_bottom: Tuple[int, ...]) -> float:
        return sum(v_costs[row_top[c], row_bottom[c]] for c in range(gs)) / gs
    
    def compute_border_cost(perm: Tuple[int, ...]) -> float:
        """Border penalty for top and bottom rows."""
        cost = 0.0
        top_row = rows[perm[0]]
        bottom_row = rows[perm[-1]]
        for c in range(gs):
            cost += get_border_penalty(top_row[c], 0, c, border_scores)
            cost += get_border_penalty(bottom_row[c], gs-1, c, border_scores)
        return cost
    
    best_order = None
    best_cost = float('inf')
    
    for perm in permutations(range(gs)):
        # Vertical seam costs
        cost = sum(row_vertical_cost(rows[perm[i]], rows[perm[i+1]]) for i in range(gs-1))
        # Border costs
        cost += compute_border_cost(perm)
        
        if cost < best_cost:
            best_cost = cost
            best_order = perm
    
    # Build board
    board = {}
    for row_idx, orig_idx in enumerate(best_order):
        for col_idx, pid in enumerate(rows[orig_idx]):
            board[(row_idx, col_idx)] = pid
    
    return board


def compute_board_score(board: Dict[Tuple[int, int], int], h_costs: np.ndarray, 
                        v_costs: np.ndarray, border_scores: Dict) -> float:
    """Compute total board score (seam + border)."""
    total = 0.0
    count = 0
    gs = GRID_SIZE
    
    for r in range(gs):
        for c in range(gs):
            pid = board[(r, c)]
            total += get_border_penalty(pid, r, c, border_scores)
            
            if c < gs - 1:
                total += h_costs[pid, board[(r, c + 1)]]
                count += 1
            if r < gs - 1:
                total += v_costs[pid, board[(r + 1, c)]]
                count += 1
    
    return total / count if count > 0 else 0.0


def swap_delta(board: Dict, a: Tuple[int, int], b: Tuple[int, int],
               h_costs: np.ndarray, v_costs: np.ndarray) -> float:
    """
    Compute change in seam cost when swapping pieces at positions a and b.
    Returns delta (negative = improvement).
    """
    gs = GRID_SIZE
    
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        if c > 0: neighbors.append(((r, c-1), 'h', 'left'))
        if c < gs-1: neighbors.append(((r, c+1), 'h', 'right'))
        if r > 0: neighbors.append(((r-1, c), 'v', 'top'))
        if r < gs-1: neighbors.append(((r+1, c), 'v', 'bottom'))
        return neighbors
    
    old_cost = 0.0
    new_cost = 0.0
    
    pid_a = board[a]
    pid_b = board[b]
    
    # Compute cost change for neighbors of a
    for n_pos, direction, side in get_neighbors(a):
        if n_pos == b:
            continue  # Skip direct neighbor (handled separately)
        n_pid = board[n_pos]
        if direction == 'h':
            if side == 'left':
                old_cost += h_costs[n_pid, pid_a]
                new_cost += h_costs[n_pid, pid_b]
            else:
                old_cost += h_costs[pid_a, n_pid]
                new_cost += h_costs[pid_b, n_pid]
        else:
            if side == 'top':
                old_cost += v_costs[n_pid, pid_a]
                new_cost += v_costs[n_pid, pid_b]
            else:
                old_cost += v_costs[pid_a, n_pid]
                new_cost += v_costs[pid_b, n_pid]
    
    # Compute cost change for neighbors of b
    for n_pos, direction, side in get_neighbors(b):
        if n_pos == a:
            continue
        n_pid = board[n_pos]
        if direction == 'h':
            if side == 'left':
                old_cost += h_costs[n_pid, pid_b]
                new_cost += h_costs[n_pid, pid_a]
            else:
                old_cost += h_costs[pid_b, n_pid]
                new_cost += h_costs[pid_a, n_pid]
        else:
            if side == 'top':
                old_cost += v_costs[n_pid, pid_b]
                new_cost += v_costs[n_pid, pid_a]
            else:
                old_cost += v_costs[pid_b, n_pid]
                new_cost += v_costs[pid_a, n_pid]
    
    # Handle direct edge between a and b if adjacent
    ra, ca = a
    rb, cb = b
    if ra == rb and abs(ca - cb) == 1:
        # Horizontal neighbors
        if ca < cb:
            old_cost += h_costs[pid_a, pid_b]
            new_cost += h_costs[pid_b, pid_a]
        else:
            old_cost += h_costs[pid_b, pid_a]
            new_cost += h_costs[pid_a, pid_b]
    elif ca == cb and abs(ra - rb) == 1:
        # Vertical neighbors
        if ra < rb:
            old_cost += v_costs[pid_a, pid_b]
            new_cost += v_costs[pid_b, pid_a]
        else:
            old_cost += v_costs[pid_b, pid_a]
            new_cost += v_costs[pid_a, pid_b]
    
    return new_cost - old_cost


def refine_board(board: Dict[Tuple[int, int], int], h_costs: np.ndarray,
                 v_costs: np.ndarray, max_swaps: int = MAX_REFINE_SWAPS) -> Dict:
    """
    Random pair swap hill-climb refinement.
    Only accepts swaps that improve the score.
    """
    import random
    
    board = dict(board)
    positions = list(board.keys())
    
    for _ in range(max_swaps):
        a, b = random.sample(positions, 2)
        delta = swap_delta(board, a, b, h_costs, v_costs)
        
        if delta < -1e-9:
            board[a], board[b] = board[b], board[a]
    
    return board


def greedy_fallback(h_costs: np.ndarray, v_costs: np.ndarray, 
                    border_scores: Dict) -> Dict[Tuple[int, int], int]:
    """Greedy row-major placement as fallback."""
    board = {}
    used = set()
    gs = GRID_SIZE
    
    for r in range(gs):
        for c in range(gs):
            best_pid = None
            best_cost = float('inf')
            
            for pid in range(N_PIECES):
                if pid in used:
                    continue
                
                cost = get_border_penalty(pid, r, c, border_scores)
                if c > 0:
                    cost += h_costs[board[(r, c-1)], pid]
                if r > 0:
                    cost += v_costs[board[(r-1, c)], pid]
                
                if cost < best_cost:
                    best_cost = cost
                    best_pid = pid
            
            board[(r, c)] = best_pid
            used.add(best_pid)
    
    return board


def solve_heuristic_8x8(
    artifacts: Dict[int, dict],
    verbose: bool = True
) -> Tuple[Dict[Tuple[int, int], int], Tuple[int, ...], float]:
    """
    Solve 8x8 puzzle using row-based assembly with border constraints.
    
    Algorithm:
    1. Build candidate rows using beam search (horizontal costs)
    2. Select 8 disjoint rows greedily
    3. Stack rows optimally (vertical costs + border penalties)
    4. Refine with random swap hill-climb
    """
    start_time = time.time()
    
    # Extract pieces
    pieces = {i: artifacts[i]['rgb'] for i in range(N_PIECES)}
    
    if verbose:
        print("Computing edge costs...")
    h_costs, v_costs, border_scores = compute_edge_costs(pieces)
    
    if verbose:
        print("Building candidate rows...")
    rows = build_candidate_rows(h_costs, border_scores)
    
    if verbose:
        print(f"  Generated {len(rows)} candidate rows")
    
    # Select disjoint rows
    selected_rows = select_disjoint_rows(rows)
    
    if selected_rows is None:
        if verbose:
            print("  Could not find 8 disjoint rows, using fallback...")
        board = greedy_fallback(h_costs, v_costs, border_scores)
    else:
        if verbose:
            print("  Selected 8 disjoint rows")
            print("Stacking rows optimally...")
        board = stack_rows_optimally(selected_rows, v_costs, border_scores)
    
    initial_score = compute_board_score(board, h_costs, v_costs, border_scores)
    if verbose:
        print(f"  Initial score: {initial_score:.4f}")
    
    # Refinement
    if verbose:
        print("Refining with hill-climb...")
    board = refine_board(board, h_costs, v_costs)
    
    final_score = compute_board_score(board, h_costs, v_costs, border_scores)
    elapsed = time.time() - start_time
    
    # Convert to arrangement
    arrangement = tuple(board[(r, c)] for r in range(GRID_SIZE) for c in range(GRID_SIZE))
    
    if verbose:
        print(f"Done: {elapsed:.1f}s, score={final_score:.4f}")
    
    return board, arrangement, final_score
