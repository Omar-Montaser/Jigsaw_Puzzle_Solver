"""
8x8 Puzzle Solver - Final Version

Combines the best approaches from 02_final_solver with our artifact pipeline:
- LAB color space + gradient dissimilarity (from 02_final_solver)
- Best Buddy detection for confident pairs
- A*-inspired region growing with ambiguity filtering
- Border variance scoring for orientation
- Uses our artifact pipeline for preprocessing

Compatible with accuracy_utils for evaluation.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import deque
from scipy.optimize import linear_sum_assignment

from .seam_cost import validate_artifacts


# =============================================================================
# DISSIMILARITY COMPUTATION (from 02_final_solver)
# =============================================================================

def compute_nssd(edge1: np.ndarray, edge2: np.ndarray) -> float:
    """
    Compute Normalized Sum of Squared Differences between two edges.
    NSSD = SSD / (std1 * std2) to handle varying intensities.
    """
    ssd = np.sum((edge1.astype(float) - edge2.astype(float)) ** 2)
    std1 = np.std(edge1)
    std2 = np.std(edge2)
    
    if std1 < 1.0 or std2 < 1.0:
        return ssd / edge1.size
    
    return ssd / (std1 * std2 * edge1.size)


def compute_dissimilarity_matrices(artifacts: Dict[int, dict], use_gradient: bool = True):
    """
    Compute dissimilarity matrices using LAB color space + gradient.
    
    Returns:
        h_dis: h_dis[i,j] = cost of placing piece j to the RIGHT of piece i
        v_dis: v_dis[i,j] = cost of placing piece j BELOW piece i
    """
    n = len(artifacts)
    h_dis = np.zeros((n, n))
    v_dis = np.zeros((n, n))
    
    # Precompute LAB, gray, and gradients from artifacts
    pieces_lab = []
    pieces_gray = []
    pieces_grad_x = []
    pieces_grad_y = []
    
    for i in range(n):
        rgb = artifacts[i]['rgb']
        gray = artifacts[i]['gray']
        
        # Convert BGR to LAB
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float64)
        pieces_lab.append(lab)
        pieces_gray.append(gray.astype(np.float64))
        
        if use_gradient:
            grad_x = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
            pieces_grad_x.append(grad_x)
            pieces_grad_y.append(grad_y)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                h_dis[i, j] = np.inf
                v_dis[i, j] = np.inf
                continue
            
            # Horizontal: right edge of i with left edge of j
            right_edge = pieces_lab[i][:, -1, :]
            left_edge = pieces_lab[j][:, 0, :]
            diff = right_edge - left_edge
            lab_cost = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))
            
            nssd = compute_nssd(pieces_gray[i][:, -1], pieces_gray[j][:, 0])
            h_dis[i, j] = lab_cost + 0.05 * nssd
            
            if use_gradient:
                grad_cost = np.mean(np.abs(pieces_grad_x[i][:, -1] - pieces_grad_x[j][:, 0]))
                h_dis[i, j] += 0.1 * grad_cost
            
            # Vertical: bottom edge of i with top edge of j
            bottom_edge = pieces_lab[i][-1, :, :]
            top_edge = pieces_lab[j][0, :, :]
            diff = bottom_edge - top_edge
            lab_cost = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))
            
            nssd = compute_nssd(pieces_gray[i][-1, :], pieces_gray[j][0, :])
            v_dis[i, j] = lab_cost + 0.05 * nssd
            
            if use_gradient:
                grad_cost = np.mean(np.abs(pieces_grad_y[i][-1, :] - pieces_grad_y[j][0, :]))
                v_dis[i, j] += 0.1 * grad_cost
    
    return h_dis, v_dis


# =============================================================================
# BORDER VARIANCE (for orientation detection)
# =============================================================================

def compute_edge_variance(artifacts: Dict[int, dict]) -> Dict[int, dict]:
    """
    Compute edge variance for each piece.
    Low variance on an edge suggests it was at the image border.
    """
    variances = {}
    for i, art in artifacts.items():
        rgb = art['rgb']
        variances[i] = {
            'L': np.var(rgb[:, 0, :]),
            'R': np.var(rgb[:, -1, :]),
            'T': np.var(rgb[0, :, :]),
            'B': np.var(rgb[-1, :, :]),
        }
    return variances


def compute_border_score(arr: np.ndarray, variances: dict, grid_size: int) -> float:
    """
    Compute score based on edge variance at all border positions.
    Lower score = more likely correct arrangement.
    """
    score = 0.0
    
    # Top row - should have low T variance
    for c in range(grid_size):
        p = arr[0, c]
        if p >= 0:
            score += variances[p]['T']
    
    # Bottom row - should have low B variance
    for c in range(grid_size):
        p = arr[grid_size - 1, c]
        if p >= 0:
            score += variances[p]['B']
    
    # Left column - should have low L variance
    for r in range(grid_size):
        p = arr[r, 0]
        if p >= 0:
            score += variances[p]['L']
    
    # Right column - should have low R variance
    for r in range(grid_size):
        p = arr[r, grid_size - 1]
        if p >= 0:
            score += variances[p]['R']
    
    return score


# =============================================================================
# BEST BUDDY DETECTION
# =============================================================================

def find_best_buddies(h_dis: np.ndarray, v_dis: np.ndarray) -> Tuple[Set, Set]:
    """
    Find Best Buddies - pairs of pieces that mutually prefer each other.
    """
    n = h_dis.shape[0]
    
    best_right = np.argmin(h_dis, axis=1)
    best_left = np.argmin(h_dis, axis=0)
    best_below = np.argmin(v_dis, axis=1)
    best_above = np.argmin(v_dis, axis=0)
    
    h_buddies = set()
    for i in range(n):
        j = best_right[i]
        if best_left[j] == i:
            h_buddies.add((i, j))
    
    v_buddies = set()
    for i in range(n):
        j = best_below[i]
        if best_above[j] == i:
            v_buddies.add((i, j))
    
    return h_buddies, v_buddies


# =============================================================================
# SOLUTION COST
# =============================================================================

def compute_solution_cost(arr: np.ndarray, h_dis: np.ndarray, v_dis: np.ndarray) -> float:
    """Compute total boundary cost of a solution."""
    grid_size = arr.shape[0]
    cost = 0.0
    
    # Horizontal seams
    for r in range(grid_size):
        for c in range(grid_size - 1):
            if arr[r, c] >= 0 and arr[r, c + 1] >= 0:
                cost += h_dis[arr[r, c], arr[r, c + 1]]
    
    # Vertical seams
    for r in range(grid_size - 1):
        for c in range(grid_size):
            if arr[r, c] >= 0 and arr[r + 1, c] >= 0:
                cost += v_dis[arr[r, c], arr[r + 1, c]]
    
    return cost



# =============================================================================
# A* REGION GROWING
# =============================================================================

def get_empty_neighbors(grid: np.ndarray, placed: Set[Tuple[int, int]], grid_size: int) -> Set[Tuple[int, int]]:
    """Get frontier cells adjacent to placed pieces."""
    neighbors = set()
    for (r, c) in placed:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] == -1:
                neighbors.add((nr, nc))
    return neighbors


def astar_grow(
    h_dis: np.ndarray,
    v_dis: np.ndarray,
    h_buddy_set: Set[Tuple[int, int]],
    v_buddy_set: Set[Tuple[int, int]],
    seed_pieces: List[int],
    seed_positions: List[Tuple[int, int]],
    grid_size: int = 8,
) -> np.ndarray:
    """
    A*-inspired region growing from seed pieces.
    Prioritizes buddy matches and low-ambiguity placements.
    """
    n = grid_size * grid_size
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    remaining = set(range(n)) - set(seed_pieces)
    placed = set()
    
    for piece, pos in zip(seed_pieces, seed_positions):
        grid[pos] = piece
        placed.add(pos)
    
    while remaining:
        frontier = get_empty_neighbors(grid, placed, grid_size)
        if not frontier:
            break
        
        candidates = []
        
        for (r, c) in frontier:
            slot_candidates = []
            
            for p in remaining:
                cost = 0.0
                count = 0
                is_buddy = False
                
                # Left neighbor
                if c > 0 and grid[r, c - 1] >= 0:
                    left = grid[r, c - 1]
                    cost += h_dis[left, p]
                    count += 1
                    if (left, p) in h_buddy_set:
                        is_buddy = True
                
                # Right neighbor
                if c < grid_size - 1 and grid[r, c + 1] >= 0:
                    right = grid[r, c + 1]
                    cost += h_dis[p, right]
                    count += 1
                
                # Top neighbor
                if r > 0 and grid[r - 1, c] >= 0:
                    top = grid[r - 1, c]
                    cost += v_dis[top, p]
                    count += 1
                    if (top, p) in v_buddy_set:
                        is_buddy = True
                
                # Bottom neighbor
                if r < grid_size - 1 and grid[r + 1, c] >= 0:
                    bottom = grid[r + 1, c]
                    cost += v_dis[p, bottom]
                    count += 1
                
                if count > 0:
                    avg_cost = cost / count
                    slot_candidates.append((avg_cost, p, is_buddy))
            
            if len(slot_candidates) >= 2:
                slot_candidates.sort()
                best_cost, best_p, is_buddy = slot_candidates[0]
                second_cost = slot_candidates[1][0]
                ambiguity = best_cost / second_cost if second_cost > 0 else 0
                
                candidates.append({
                    'pos': (r, c),
                    'piece': best_p,
                    'cost': best_cost,
                    'ambiguity': ambiguity,
                    'is_buddy': is_buddy,
                })
            elif slot_candidates:
                best_cost, best_p, is_buddy = slot_candidates[0]
                candidates.append({
                    'pos': (r, c),
                    'piece': best_p,
                    'cost': best_cost,
                    'ambiguity': 0,
                    'is_buddy': is_buddy,
                })
        
        if not candidates:
            break
        
        # Prioritize: buddies first, then low ambiguity, then low cost
        candidates.sort(key=lambda x: (not x['is_buddy'], x['ambiguity'], x['cost']))
        
        best = candidates[0]
        r, c = best['pos']
        grid[r, c] = best['piece']
        remaining.remove(best['piece'])
        placed.add((r, c))
    
    return grid


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_8x8_final(artifacts: Dict[int, dict], verbose: bool = True) -> Tuple[dict, list, float]:
    """
    Solve 8x8 puzzle using LAB + gradient dissimilarity with A* region growing.
    
    Args:
        artifacts: dict of piece_id -> artifact dict with 'rgb', 'gray', 'edges', 'blur'
        verbose: print progress
    
    Returns:
        Tuple of (board dict, arrangement list, final score)
    """
    validate_artifacts(artifacts)
    grid_size = 8
    n = 64
    
    if verbose:
        print("=" * 60)
        print("8x8 Puzzle Solver (LAB + Gradient + A* Region Growing)")
        print("=" * 60)
    
    # Compute dissimilarity matrices
    if verbose:
        print("\n[1] Computing dissimilarity matrices...")
    h_dis, v_dis = compute_dissimilarity_matrices(artifacts, use_gradient=True)
    
    # Find best buddies
    if verbose:
        print("\n[2] Finding best buddies...")
    h_buddies, v_buddies = find_best_buddies(h_dis, v_dis)
    h_buddy_set = set(h_buddies)
    v_buddy_set = set(v_buddies)
    if verbose:
        print(f"    Found {len(h_buddies)} horizontal + {len(v_buddies)} vertical buddies")
    
    # Compute edge variances for border scoring
    variances = compute_edge_variance(artifacts)
    
    # Collect all solutions
    all_solutions = []
    seen_keys = set()
    
    def add_solution(arr, method):
        if (arr < 0).any():
            return
        key = tuple(arr.flatten())
        if key not in seen_keys:
            seen_keys.add(key)
            cost = compute_solution_cost(arr, h_dis, v_dis)
            border = compute_border_score(arr, variances, grid_size)
            all_solutions.append({
                'arr': arr.copy(),
                'cost': cost,
                'border': border,
                'method': method,
            })
    
    # Strategy 1: A* grow from strongest buddy pairs
    if verbose:
        print("\n[3] A* region growing from buddy pairs...")
    
    buddy_pairs = []
    for i, j in h_buddy_set:
        buddy_pairs.append((h_dis[i, j], i, j, 'h'))
    for i, j in v_buddy_set:
        buddy_pairs.append((v_dis[i, j], i, j, 'v'))
    buddy_pairs.sort()
    
    for cost, p1, p2, direction in buddy_pairs[:15]:
        if direction == 'h':
            seed_pieces = [p1, p2]
            seed_positions = [(3, 3), (3, 4)]
        else:
            seed_pieces = [p1, p2]
            seed_positions = [(3, 3), (4, 3)]
        
        arr = astar_grow(h_dis, v_dis, h_buddy_set, v_buddy_set, seed_pieces, seed_positions, grid_size)
        add_solution(arr, f'astar_buddy_{direction}')
    
    # Strategy 2: Row-wise greedy from different starts
    if verbose:
        print("\n[4] Row-wise greedy assembly...")
    
    for start in range(min(16, n)):
        arr = np.full((grid_size, grid_size), -1, dtype=int)
        used = {start}
        arr[0, 0] = start
        
        for r in range(grid_size):
            for c in range(grid_size):
                if arr[r, c] != -1:
                    continue
                
                best_p, best_c = None, np.inf
                for p in range(n):
                    if p in used:
                        continue
                    cost = 0.0
                    buddy_bonus = 0
                    
                    if c > 0:
                        left = arr[r, c - 1]
                        cost += h_dis[left, p]
                        if (left, p) in h_buddy_set:
                            buddy_bonus += 1
                    if r > 0:
                        top = arr[r - 1, c]
                        cost += v_dis[top, p]
                        if (top, p) in v_buddy_set:
                            buddy_bonus += 1
                    
                    adj = cost / (1 + buddy_bonus * 0.5)
                    if adj < best_c:
                        best_c, best_p = adj, p
                
                if best_p is not None:
                    arr[r, c] = best_p
                    used.add(best_p)
        
        add_solution(arr, f'greedy_start_{start}')
    
    if verbose:
        print(f"    Generated {len(all_solutions)} candidate solutions")
    
    # Select best solution
    if verbose:
        print("\n[5] Selecting best solution...")
    
    if not all_solutions:
        # Fallback: return identity
        arr = np.arange(n).reshape(grid_size, grid_size)
        board = {(r, c): arr[r, c] for r in range(grid_size) for c in range(grid_size)}
        arrangement = list(arr.flatten())
        return board, arrangement, float('inf')
    
    # Rank by cost + small border penalty
    all_solutions.sort(key=lambda x: x['cost'] + 0.001 * x['border'])
    best = all_solutions[0]
    
    if verbose:
        print(f"    Best method: {best['method']}")
        print(f"    Cost: {best['cost']:.2f}, Border: {best['border']:.2f}")
    
    # Convert to board dict
    arr = best['arr']
    board = {(r, c): int(arr[r, c]) for r in range(grid_size) for c in range(grid_size)}
    arrangement = [int(arr[r, c]) for r in range(grid_size) for c in range(grid_size)]
    
    return board, arrangement, best['cost']