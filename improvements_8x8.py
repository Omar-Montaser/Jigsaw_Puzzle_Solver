"""
Improved 8x8 Puzzle Solver - Complete Pipeline

Implements:
- Phase 0: Precompute all descriptors and compatibility
- Phase 1: Confident pair detection & mutual locking
- Phase 2: Superpiece assembly (CSP/Hungarian)
- Phase 3: Global refinement (swaps, simulated annealing, KL-style)
- Phase 4: Diagnostics & fallback

All weights and thresholds configurable. Lower score = better match.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import heapq
import random
from scipy.optimize import linear_sum_assignment


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SolverConfig:
    """All configurable parameters."""
    # Edge descriptor weights (must sum to 1.0)
    # Pixel-only baseline (matches direct_pixel_solver.py behavior)
    pixel_weight: float = 1.0
    color_weight: float = 0.0
    lbp_weight: float = 0.0
    curv_weight: float = 0.0
    centroid_weight: float = 0.0
    
    # Confident pair thresholds
    # Keep these conservative; locking is diagnostic-only in this file.
    margin_threshold: float = 0.50
    mutual_best_conf_ratio: float = 0.50
    very_confident_ratio: float = 0.10
    
    # Simulated annealing
    sa_t0: float = 0.15
    sa_alpha: float = 0.9995
    sa_max_iters: int = 5000
    sa_time_limit: float = 30.0  # seconds
    
    # Beam search fallback
    beam_seed_width: int = 200
    
    # Acceptable score threshold
    acceptable_threshold: float = 0.12
    
    # Strip width for pixel extraction (1px = sharp edges, more works for some puzzles)
    strip_width: int = 1
    
    # Histogram bins
    color_bins: int = 16
    lbp_bins: int = 8
    
    # Resampling length for signatures
    resample_len: int = 128
    
    # Refinement passes
    refinement_passes: int = 3
    
    # Debug output
    debug_dir: str = "./debug"
    verbose: bool = True


# =============================================================================
# PHASE 0: DESCRIPTOR COMPUTATION
# =============================================================================

class EdgeDescriptors:
    """Precomputed descriptors for a single edge."""
    
    def __init__(self, piece: np.ndarray, edge: str, config: SolverConfig):
        self.edge = edge
        strip = self._get_strip(piece, edge, config.strip_width)
        
        # Compute all descriptors
        self.pixel_strip = self._compute_pixel_strip(strip)
        self.color_hist = self._compute_color_hist(strip, config.color_bins)
        self.lbp_hist = self._compute_lbp_hist(strip, config.lbp_bins)
        self.curvature = self._compute_curvature(strip, config.resample_len)
        self.centroid_radial = self._compute_centroid_radial(strip, config.resample_len)
    
    def _get_strip(self, piece: np.ndarray, edge: str, width: int) -> np.ndarray:
        # IMPORTANT ORIENTATION RULE:
        # For pixel-strip matching, we want the strip ordered as "boundary -> inward".
        # This makes a 3-pixel strip comparable across complementary edges.
        if edge == 'top':
            # boundary is row 0, inward is increasing y
            return piece[:width, :, :] if len(piece.shape) == 3 else piece[:width, :]
        elif edge == 'bottom':
            # boundary is last row, inward is decreasing y
            strip = piece[-width:, :, :] if len(piece.shape) == 3 else piece[-width:, :]
            return strip[::-1]
        elif edge == 'left':
            # boundary is col 0, inward is increasing x
            return piece[:, :width, :] if len(piece.shape) == 3 else piece[:, :width]
        elif edge == 'right':
            # boundary is last col, inward is decreasing x
            strip = piece[:, -width:, :] if len(piece.shape) == 3 else piece[:, -width:]
            return strip[:, ::-1]
        raise ValueError(f"Unknown edge: {edge}")
    
    def _compute_pixel_strip(self, strip: np.ndarray) -> np.ndarray:
        """Flattened, normalized pixel values."""
        pixels = strip.flatten().astype(np.float32) / 255.0
        return pixels
    
    def _compute_color_hist(self, strip: np.ndarray, bins: int) -> np.ndarray:
        """Color histogram, L1 normalized."""
        if len(strip.shape) == 3:
            hists = []
            for c in range(strip.shape[2]):
                h, _ = np.histogram(strip[:, :, c].flatten(), bins=bins, range=(0, 256))
                hists.append(h)
            hist = np.concatenate(hists).astype(np.float64)
        else:
            hist, _ = np.histogram(strip.flatten(), bins=bins, range=(0, 256))
            hist = hist.astype(np.float64)
        
        total = np.sum(hist)
        if total > 0:
            hist /= total
        return hist
    
    def _compute_lbp_hist(self, strip: np.ndarray, bins: int) -> np.ndarray:
        """LBP histogram, L1 normalized."""
        if len(strip.shape) == 3:
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        else:
            gray = strip.copy()
        
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return np.zeros(bins)
        
        h, w = gray.shape
        lbp_codes = []
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = gray[y, x]
                pattern = 0
                neighbors = [
                    (y-1, x-1), (y-1, x), (y-1, x+1),
                    (y, x+1), (y+1, x+1), (y+1, x),
                    (y+1, x-1), (y, x-1)
                ]
                for i, (ny, nx) in enumerate(neighbors):
                    if gray[ny, nx] >= center:
                        pattern |= (1 << i)
                lbp_codes.append(bin(pattern).count('1') % bins)
        
        if not lbp_codes:
            return np.zeros(bins)
        
        hist, _ = np.histogram(lbp_codes, bins=bins, range=(0, bins))
        hist = hist.astype(np.float64)
        total = np.sum(hist)
        if total > 0:
            hist /= total
        return hist
    
    def _compute_curvature(self, strip: np.ndarray, resample_len: int) -> np.ndarray:
        """Curvature signature (intensity-based proxy)."""
        if len(strip.shape) == 3:
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        else:
            gray = strip.copy()
        
        h, w = gray.shape
        if h > w:
            profile = gray.mean(axis=1)
        else:
            profile = gray.mean(axis=0)
        
        if len(profile) < 3:
            return np.zeros(resample_len)
        
        gradient = np.gradient(profile.astype(np.float64))
        curvature = np.gradient(gradient)
        curvature = self._resample(curvature, resample_len)
        
        norm = np.linalg.norm(curvature)
        if norm > 1e-10:
            curvature /= norm
        return curvature
    
    def _compute_centroid_radial(self, strip: np.ndarray, resample_len: int) -> np.ndarray:
        """Centroid-radial distance signature."""
        if len(strip.shape) == 3:
            h, w, _ = strip.shape
        else:
            h, w = strip.shape
        
        centroid = np.array([w / 2, h / 2])
        
        if h > w:
            points = np.array([[w/2, y] for y in range(h)])
        else:
            points = np.array([[x, h/2] for x in range(w)])
        
        if len(points) == 0:
            return np.zeros(resample_len)
        
        distances = np.linalg.norm(points - centroid, axis=1)
        distances = self._resample(distances, resample_len)
        
        max_dist = np.max(distances)
        if max_dist > 1e-10:
            distances /= max_dist
        return distances
    
    def _resample(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        if len(signal) == 0:
            return np.zeros(target_len)
        if len(signal) == target_len:
            return signal
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, signal)


class CompatibilityMatrix:
    """
    Precomputed compatibility scores between all edge pairs.
    """
    
    def __init__(self, pieces: Dict[int, np.ndarray], config: SolverConfig):
        self.config = config
        self.piece_ids = sorted(pieces.keys())
        self.n_pieces = len(self.piece_ids)
        self.edges = ['top', 'bottom', 'left', 'right']
        self.complementary = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        
        # Compute all descriptors
        print("  Computing edge descriptors...")
        self.descriptors: Dict[Tuple[int, str], EdgeDescriptors] = {}
        for pid in self.piece_ids:
            for edge in self.edges:
                self.descriptors[(pid, edge)] = EdgeDescriptors(pieces[pid], edge, config)
        
        # Compute all compatibilities
        print("  Computing pairwise compatibility...")
        # compatibility[(pid_a, edge_a, pid_b, edge_b)] = score
        self.compatibility: Dict[Tuple[int, str, int, str], float] = {}
        
        # best_match[(pid, edge)] = (best_pid, best_edge, best_score)
        self.best_match: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        
        # second_best_match[(pid, edge)] = (pid, edge, score)
        self.second_best: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        
        self._compute_all()
    
    def _chi_squared(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Chi-squared distance between histograms."""
        denom = h1 + h2
        mask = denom > 1e-10
        if not np.any(mask):
            return 0.0
        diff_sq = (h1[mask] - h2[mask]) ** 2
        chi2 = np.sum(diff_sq / denom[mask])
        return min(chi2 / 2.0, 1.0)
    
    def _compute_edge_score(self, desc1: EdgeDescriptors, desc2: EdgeDescriptors) -> float:
        """Compute combined score between two edges (lower = better)."""
        cfg = self.config
        
        # Pixel MAE
        p1, p2 = desc1.pixel_strip, desc2.pixel_strip
        min_len = min(len(p1), len(p2))
        if min_len > 0:
            score_pixel = np.mean(np.abs(p1[:min_len] - p2[:min_len]))
        else:
            score_pixel = 1.0
        
        # Color histogram chi-squared
        score_color = self._chi_squared(desc1.color_hist, desc2.color_hist)
        
        # LBP histogram chi-squared
        score_lbp = self._chi_squared(desc1.lbp_hist, desc2.lbp_hist)
        
        # Curvature complementarity (reversed)
        curv1 = desc1.curvature
        curv2_rev = desc2.curvature[::-1]
        curv_sum = curv1 + curv2_rev
        score_curv = np.linalg.norm(curv_sum) / 2.0
        score_curv = min(score_curv, 1.0)
        
        # Centroid radial (reversed)
        rad1 = desc1.centroid_radial
        rad2_rev = desc2.centroid_radial[::-1]
        rad_diff = rad1 - rad2_rev
        score_centroid = np.linalg.norm(rad_diff) / np.sqrt(len(rad1))
        score_centroid = min(score_centroid, 1.0)
        
        # Combined score
        combined = (
            cfg.pixel_weight * score_pixel +
            cfg.color_weight * score_color +
            cfg.lbp_weight * score_lbp +
            cfg.curv_weight * score_curv +
            cfg.centroid_weight * score_centroid
        )
        
        return combined
    
    def _compute_all(self):
        """Compute all pairwise edge compatibilities."""
        total = self.n_pieces * 4
        count = 0
        
        for pid_a in self.piece_ids:
            for edge_a in self.edges:
                comp_edge = self.complementary[edge_a]
                candidates = []
                
                for pid_b in self.piece_ids:
                    if pid_a == pid_b:
                        continue
                    
                    desc_a = self.descriptors[(pid_a, edge_a)]
                    desc_b = self.descriptors[(pid_b, comp_edge)]
                    score = self._compute_edge_score(desc_a, desc_b)
                    
                    self.compatibility[(pid_a, edge_a, pid_b, comp_edge)] = score
                    candidates.append((score, pid_b, comp_edge))
                
                # Sort by score
                candidates.sort()
                
                if len(candidates) >= 1:
                    self.best_match[(pid_a, edge_a)] = (candidates[0][1], candidates[0][2], candidates[0][0])
                if len(candidates) >= 2:
                    self.second_best[(pid_a, edge_a)] = (candidates[1][1], candidates[1][2], candidates[1][0])
                else:
                    self.second_best[(pid_a, edge_a)] = (None, None, float('inf'))
                
                count += 1
                if count % 64 == 0:
                    print(f"    Progress: {count}/{total}")
    
    def get_score(self, pid_a: int, edge_a: str, pid_b: int, edge_b: str) -> float:
        """Get compatibility score."""
        key = (pid_a, edge_a, pid_b, edge_b)
        v = self.compatibility.get(key)
        if v is None:
            # Should be rare (we precompute everything), but avoid inf/nan poisoning refinement.
            desc_a = self.descriptors[(pid_a, edge_a)]
            desc_b = self.descriptors[(pid_b, edge_b)]
            v = self._compute_edge_score(desc_a, desc_b)
            self.compatibility[key] = v
        return v
    
    def get_horizontal_score(self, left_pid: int, right_pid: int) -> float:
        """Score for placing right_pid to the right of left_pid."""
        return self.get_score(left_pid, 'right', right_pid, 'left')
    
    def get_vertical_score(self, top_pid: int, bottom_pid: int) -> float:
        """Score for placing bottom_pid below top_pid."""
        return self.get_score(top_pid, 'bottom', bottom_pid, 'top')


# =============================================================================
# PHASE 1: CONFIDENT PAIR DETECTION
# =============================================================================

@dataclass
class LockedPair:
    """A confidently matched edge pair."""
    piece_a: int
    edge_a: str
    piece_b: int
    edge_b: str
    score: float
    confidence_ratio: float
    margin: float


class ConfidentPairDetector:
    """Detect and lock confident edge pairs."""
    
    def __init__(self, compat: CompatibilityMatrix, config: SolverConfig):
        self.compat = compat
        self.config = config
        self.locked_pairs: List[LockedPair] = []
        self.superpieces: Dict[int, Set[int]] = {}  # superpiece_id -> set of piece_ids
        self.piece_to_super: Dict[int, int] = {}  # piece_id -> superpiece_id
    
    def detect_pairs(self) -> List[LockedPair]:
        """
        Detect confident pairs based on:
        1. Mutual best match with conf_ratio < 0.9
        2. Or margin > threshold
        3. Or conf_ratio < 0.6 (very confident)
        """
        cfg = self.config
        seen = set()
        
        for pid_a in self.compat.piece_ids:
            for edge_a in self.compat.edges:
                best = self.compat.best_match.get((pid_a, edge_a))
                second = self.compat.second_best.get((pid_a, edge_a))
                
                if best is None or best[0] is None:
                    continue
                
                pid_b, edge_b, best_score = best
                second_score = second[2] if second and second[0] is not None else float('inf')
                
                # Compute metrics
                conf_ratio = best_score / (second_score + 1e-9)
                margin = second_score - best_score
                
                # Check if B's best match is A (mutual)
                b_best = self.compat.best_match.get((pid_b, edge_b))
                is_mutual = b_best is not None and b_best[0] == pid_a and b_best[1] == edge_a
                
                # Locking conditions
                should_lock = False
                
                # Condition 1: Mutual best with good confidence
                if is_mutual and conf_ratio < cfg.mutual_best_conf_ratio:
                    should_lock = True
                
                # Condition 2: Large margin
                if margin > cfg.margin_threshold:
                    should_lock = True
                
                # Condition 3: Very confident
                if conf_ratio < cfg.very_confident_ratio:
                    should_lock = True
                
                if should_lock:
                    # Avoid duplicates
                    pair_key = tuple(sorted([(pid_a, edge_a), (pid_b, edge_b)]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        self.locked_pairs.append(LockedPair(
                            piece_a=pid_a, edge_a=edge_a,
                            piece_b=pid_b, edge_b=edge_b,
                            score=best_score,
                            confidence_ratio=conf_ratio,
                            margin=margin
                        ))
        
        print(f"  Detected {len(self.locked_pairs)} locked pairs")
        return self.locked_pairs
    
    def build_superpieces(self) -> Dict[int, Set[int]]:
        """
        Build superpieces from locked pairs using union-find.
        """
        # Union-Find
        parent = {pid: pid for pid in self.compat.piece_ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union pieces in locked pairs
        for pair in self.locked_pairs:
            union(pair.piece_a, pair.piece_b)
        
        # Group by root
        groups = defaultdict(set)
        for pid in self.compat.piece_ids:
            root = find(pid)
            groups[root].add(pid)
        
        # Assign superpiece IDs
        self.superpieces = {}
        self.piece_to_super = {}
        
        for idx, (root, members) in enumerate(groups.items()):
            self.superpieces[idx] = members
            for pid in members:
                self.piece_to_super[pid] = idx
        
        sizes = sorted([len(s) for s in self.superpieces.values()], reverse=True)
        print(f"  Created {len(self.superpieces)} superpieces")
        print(f"  Sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
        
        return self.superpieces


# =============================================================================
# PHASE 2: SUPERPIECE ASSEMBLY
# =============================================================================

class PuzzleAssembler:
    """Assemble puzzle from superpieces."""
    
    def __init__(self, compat: CompatibilityMatrix, 
                 detector: ConfidentPairDetector,
                 grid_size: int,
                 config: SolverConfig):
        self.compat = compat
        self.detector = detector
        self.grid_size = grid_size
        self.config = config
        
        # Board: (row, col) -> piece_id
        self.board: Dict[Tuple[int, int], int] = {}
    
    def evaluate_board(self, board: Optional[Dict] = None) -> float:
        """Compute average edge score for current board."""
        if board is None:
            board = self.board
        
        total = 0.0
        count = 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in board:
                    continue
                pid = board[(row, col)]
                
                # Right neighbor
                if col < self.grid_size - 1 and (row, col + 1) in board:
                    right_pid = board[(row, col + 1)]
                    total += self.compat.get_horizontal_score(pid, right_pid)
                    count += 1
                
                # Bottom neighbor
                if row < self.grid_size - 1 and (row + 1, col) in board:
                    bottom_pid = board[(row + 1, col)]
                    total += self.compat.get_vertical_score(pid, bottom_pid)
                    count += 1
        
        return total / max(count, 1)
    
    def assemble_beam_search(self, beam_width: int = 10000) -> Dict[Tuple[int, int], int]:
        """Beam search assembly."""
        import heapq
        piece_ids = self.compat.piece_ids
        n = len(piece_ids)
        
        print(f"  Beam search (width={beam_width})...")
        
        # State: (arrangement tuple, used set, cum_score, num_edges)
        initial = (tuple([None] * n), frozenset(), 0.0, 0)
        beam = [initial]
        
        for pos in range(n):
            row, col = pos // self.grid_size, pos % self.grid_size
            new_beam = []
            
            for arr, used, cum_score, num_edges in beam:
                for pid in piece_ids:
                    if pid in used:
                        continue
                    
                    edge_score = 0.0
                    edges = 0
                    
                    # Left neighbor
                    if col > 0 and arr[pos - 1] is not None:
                        edge_score += self.compat.get_horizontal_score(arr[pos - 1], pid)
                        edges += 1
                    
                    # Top neighbor
                    if row > 0 and arr[pos - self.grid_size] is not None:
                        edge_score += self.compat.get_vertical_score(arr[pos - self.grid_size], pid)
                        edges += 1
                    
                    new_arr = list(arr)
                    new_arr[pos] = pid
                    new_beam.append((
                        tuple(new_arr),
                        used | {pid},
                        cum_score + edge_score,
                        num_edges + edges
                    ))
            
            # Keep best (avoid full sort of potentially huge lists)
            beam = heapq.nsmallest(beam_width, new_beam, key=lambda s: s[2] / max(s[3], 1))
            
            if (pos + 1) % self.grid_size == 0:
                print(f"    Row {(pos + 1) // self.grid_size}: {len(new_beam)} -> {len(beam)}")
        
        # Convert to board
        best_arr = beam[0][0]
        board = {}
        for pos, pid in enumerate(best_arr):
            row, col = pos // self.grid_size, pos % self.grid_size
            board[(row, col)] = pid
        
        return board
    
    def assemble_hungarian_rows(self) -> Dict[Tuple[int, int], int]:
        """
        Build rows using Hungarian algorithm, then stack rows optimally.
        """
        print("  Hungarian row assembly...")
        piece_ids = list(self.compat.piece_ids)
        n = len(piece_ids)
        gs = self.grid_size
        
        # Phase 1: Build best rows using beam search
        print("    Phase 1: Building candidate rows...")
        row_beam = 5000
        
        row_states = [(tuple([p]), {p}, 0.0) for p in piece_ids]
        
        for col in range(1, gs):
            new_states = []
            for row, used, score in row_states:
                last = row[-1]
                for pid in piece_ids:
                    if pid in used:
                        continue
                    edge_score = self.compat.get_horizontal_score(last, pid)
                    new_states.append((row + (pid,), used | {pid}, score + edge_score))
            
            new_states.sort(key=lambda x: x[2] / col)
            row_states = new_states[:row_beam]
        
        complete_rows = [(r, s / (gs - 1)) for r, u, s in row_states]
        complete_rows.sort(key=lambda x: x[1])
        
        print(f"    Got {len(complete_rows)} candidate rows")
        
        # Phase 2: Select 8 non-overlapping rows.
        print("    Phase 2: Selecting best rows with Hungarian...")

        # The previous greedy selection often gets stuck (e.g., finds only 1 row).
        # Use a small backtracking search over the best candidates to find 8 DISJOINT rows.
        max_pool = min(len(complete_rows), 800)
        pool = complete_rows[:max_pool]

        # Precompute bitmasks for fast disjoint checks (piece ids are 0..n-1).
        row_entries: List[Tuple[int, Tuple[int, ...], float]] = []
        for row, score in pool:
            mask = 0
            for pid in row:
                mask |= (1 << int(pid))
            row_entries.append((mask, row, score))

        row_entries.sort(key=lambda t: t[2])

        piece_to_rows: Dict[int, List[int]] = {int(pid): [] for pid in piece_ids}
        for idx, (mask, row, score) in enumerate(row_entries):
            for pid in row:
                piece_to_rows[int(pid)].append(idx)

        best_rows: Optional[List[Tuple[int, ...]]] = None

        def dfs(chosen: List[int], used_mask: int) -> bool:
            nonlocal best_rows
            if len(chosen) == gs:
                best_rows = [row_entries[i][1] for i in chosen]
                return True

            remaining = [int(pid) for pid in piece_ids if not (used_mask & (1 << int(pid)))]
            if not remaining:
                return False

            def candidate_count(pid: int) -> int:
                return sum(1 for ridx in piece_to_rows[pid] if (row_entries[ridx][0] & used_mask) == 0)

            pivot = min(remaining, key=candidate_count)
            options = [ridx for ridx in piece_to_rows[pivot] if (row_entries[ridx][0] & used_mask) == 0]
            options.sort(key=lambda ridx: row_entries[ridx][2])

            for ridx in options:
                mask, _, _ = row_entries[ridx]
                if mask & used_mask:
                    continue
                chosen.append(ridx)
                if dfs(chosen, used_mask | mask):
                    return True
                chosen.pop()
            return False

        found = dfs([], 0)
        if not found or best_rows is None:
            print(f"    Warning: Could not select {gs} disjoint rows from top {max_pool}")
            self._hungarian_fell_back_to_beam = True
            return self.assemble_beam_search(10000)

        top_rows = best_rows
        
        # Phase 3: Find optimal stacking order using Hungarian
        print("    Phase 3: Optimal row ordering...")
        
        # Cost matrix: cost[i][j] = cost of putting top_rows[i] in position j
        # For now, we compute pairwise vertical costs and use them
        
        # Compute vertical compatibility between rows
        def row_vertical_cost(row_top, row_bottom):
            total = 0.0
            for c in range(gs):
                total += self.compat.get_vertical_score(row_top[c], row_bottom[c])
            return total / gs
        
        # Try all permutations for small gs (8! = 40320 is feasible)
        from itertools import permutations
        
        best_order = None
        best_cost = float('inf')
        
        for perm in permutations(range(gs)):
            cost = 0.0
            for i in range(gs - 1):
                cost += row_vertical_cost(top_rows[perm[i]], top_rows[perm[i+1]])
            if cost < best_cost:
                best_cost = cost
                best_order = perm
        
        print(f"    Best stacking cost: {best_cost:.4f}")
        
        # Build board
        board = {}
        for row_idx, orig_idx in enumerate(best_order):
            for col_idx, pid in enumerate(top_rows[orig_idx]):
                board[(row_idx, col_idx)] = pid
        
        return board
    
    def assemble(self) -> Dict[Tuple[int, int], int]:
        """Main assembly entry point."""
        self._hungarian_fell_back_to_beam = False
        # Try Hungarian first
        board1 = self.assemble_hungarian_rows()
        score1 = self.evaluate_board(board1)
        print(f"  Hungarian score: {score1:.4f}")

        # If Hungarian path already fell back to beam search, don't do a second beam run.
        if self._hungarian_fell_back_to_beam:
            self.board = board1
            return board1

        # Otherwise, also try beam search as a baseline
        board2 = self.assemble_beam_search(3000)
        score2 = self.evaluate_board(board2)
        print(f"  Beam search score: {score2:.4f}")

        # Pick best
        self.board = board1 if score1 <= score2 else board2
        return self.board


# =============================================================================
# PHASE 3: GLOBAL REFINEMENT
# =============================================================================

class PuzzleRefiner:
    """Refine puzzle solution."""
    
    def __init__(self, compat: CompatibilityMatrix, grid_size: int, config: SolverConfig):
        self.compat = compat
        self.grid_size = grid_size
        self.config = config
    
    def evaluate(self, board: Dict[Tuple[int, int], int]) -> float:
        """Compute board score."""
        total = 0.0
        count = 0
        gs = self.grid_size
        
        for row in range(gs):
            for col in range(gs):
                pid = board.get((row, col))
                if pid is None:
                    continue
                
                if col < gs - 1:
                    right = board.get((row, col + 1))
                    if right is not None:
                        total += self.compat.get_horizontal_score(pid, right)
                        count += 1
                
                if row < gs - 1:
                    bottom = board.get((row + 1, col))
                    if bottom is not None:
                        total += self.compat.get_vertical_score(pid, bottom)
                        count += 1
        
        return total / max(count, 1)

    def _total_edge_sum(self, board: Dict[Tuple[int, int], int]) -> float:
        """Sum of all neighbor edge scores for a FULL board."""
        total = 0.0
        gs = self.grid_size
        for r in range(gs):
            for c in range(gs):
                pid = board[(r, c)]
                if c < gs - 1:
                    total += self.compat.get_horizontal_score(pid, board[(r, c + 1)])
                if r < gs - 1:
                    total += self.compat.get_vertical_score(pid, board[(r + 1, c)])
        return total

    def _swap_delta_sum(self, board: Dict[Tuple[int, int], int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Delta in total edge SUM when swapping pieces at positions a and b."""
        gs = self.grid_size

        def piece_after_swap(r: int, c: int) -> int:
            if (r, c) == a:
                return board[b]
            if (r, c) == b:
                return board[a]
            return board[(r, c)]

        affected: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        def add_edges_around(pos: Tuple[int, int]):
            r, c = pos
            if c > 0:
                affected.add(((r, c - 1), (r, c)))
            if c < gs - 1:
                affected.add(((r, c), (r, c + 1)))
            if r > 0:
                affected.add(((r - 1, c), (r, c)))
            if r < gs - 1:
                affected.add(((r, c), (r + 1, c)))

        add_edges_around(a)
        add_edges_around(b)

        old_sum = 0.0
        new_sum = 0.0
        for (p1, p2) in affected:
            (r1, c1), (r2, c2) = p1, p2
            pid1_old = board[(r1, c1)]
            pid2_old = board[(r2, c2)]
            pid1_new = piece_after_swap(r1, c1)
            pid2_new = piece_after_swap(r2, c2)

            if r1 == r2:
                # Horizontal edge (left -> right)
                if c1 < c2:
                    old_sum += self.compat.get_horizontal_score(pid1_old, pid2_old)
                    new_sum += self.compat.get_horizontal_score(pid1_new, pid2_new)
                else:
                    old_sum += self.compat.get_horizontal_score(pid2_old, pid1_old)
                    new_sum += self.compat.get_horizontal_score(pid2_new, pid1_new)
            else:
                # Vertical edge (top -> bottom)
                if r1 < r2:
                    old_sum += self.compat.get_vertical_score(pid1_old, pid2_old)
                    new_sum += self.compat.get_vertical_score(pid1_new, pid2_new)
                else:
                    old_sum += self.compat.get_vertical_score(pid2_old, pid1_old)
                    new_sum += self.compat.get_vertical_score(pid2_new, pid1_new)

        return new_sum - old_sum
    
    def local_swap_refinement(self, board: Dict, passes: int = 3, samples_per_pos: int = 12) -> Dict:
        """Local swap optimization using RANDOM sampling (fast)."""
        board = dict(board)
        total_edges = 2 * self.grid_size * (self.grid_size - 1)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(total_edges, 1)
        
        print(f"  Local swap refinement (initial: {current_score:.4f})...")
        
        positions = list(board.keys())
        
        for pass_num in range(passes):
            improved = False
            improvements = 0

            # For each position, sample a few other positions Q to swap with.
            for pos1 in positions:
                for _ in range(samples_per_pos):
                    pos2 = random.choice(positions)
                    if pos1 == pos2:
                        continue

                    delta = self._swap_delta_sum(board, pos1, pos2)
                    if delta < -1e-12:
                        board[pos1], board[pos2] = board[pos2], board[pos1]
                        current_sum += delta
                        current_score = current_sum / max(total_edges, 1)
                        improved = True
                        improvements += 1
            
            print(f"    Pass {pass_num + 1}: score={current_score:.4f}, improvements={improvements}")
            
            if not improved:
                break
        
        return board

    def _block_swap_2x2(self, board: Dict[Tuple[int, int], int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Pick two random 2x2 blocks and swap them (returns top-left corners)."""
        gs = self.grid_size
        if gs < 2:
            return None

        r1 = random.randint(0, gs - 2)
        c1 = random.randint(0, gs - 2)
        r2 = random.randint(0, gs - 2)
        c2 = random.randint(0, gs - 2)
        if (r1, c1) == (r2, c2):
            return None
        return (r1, c1), (r2, c2)

    def _apply_block_swap_2x2(self, board: Dict[Tuple[int, int], int], a: Tuple[int, int], b: Tuple[int, int]) -> None:
        """Swap the 2x2 block at top-left a with the 2x2 block at top-left b."""
        (r1, c1), (r2, c2) = a, b
        coords1 = [(r1, c1), (r1, c1 + 1), (r1 + 1, c1), (r1 + 1, c1 + 1)]
        coords2 = [(r2, c2), (r2, c2 + 1), (r2 + 1, c2), (r2 + 1, c2 + 1)]
        vals1 = [board[p] for p in coords1]
        vals2 = [board[p] for p in coords2]
        for p, v in zip(coords1, vals2):
            board[p] = v
        for p, v in zip(coords2, vals1):
            board[p] = v
    
    def simulated_annealing(self, board: Dict) -> Dict:
        """Simulated annealing refinement."""
        cfg = self.config
        board = dict(board)
        total_edges = 2 * self.grid_size * (self.grid_size - 1)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(total_edges, 1)
        best_board = dict(board)
        best_score = current_score
        
        print(f"  Simulated annealing (initial: {current_score:.4f})...")
        
        T = cfg.sa_t0
        positions = list(board.keys())
        start_time = time.time()
        
        for iteration in range(cfg.sa_max_iters):
            # Check time limit
            if time.time() - start_time > cfg.sa_time_limit:
                print(f"    Time limit reached at iteration {iteration}")
                break
            
            move_type = random.random()

            # Proposal moves:
            # - random swap
            # - 2x2 block swap
            if move_type < 0.85:
                pos1, pos2 = random.sample(positions, 2)
                delta_sum = self._swap_delta_sum(board, pos1, pos2)
                board[pos1], board[pos2] = board[pos2], board[pos1]
                undo = ('swap', pos1, pos2)
            else:
                blocks = self._block_swap_2x2(board)
                if blocks is None:
                    continue
                a, b = blocks
                self._apply_block_swap_2x2(board, a, b)
                undo = ('block', a, b)

            if undo[0] == 'swap':
                delta = delta_sum / max(total_edges, 1)
                new_score = current_score + delta
            else:
                new_score = self.evaluate(board)
                delta = new_score - current_score
            
            # Accept?
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if undo[0] == 'swap':
                    current_sum += delta_sum
                if current_score < best_score:
                    best_score = current_score
                    best_board = dict(board)
            else:
                # Undo
                if undo[0] == 'swap':
                    _, p1, p2 = undo
                    board[p1], board[p2] = board[p2], board[p1]
                else:
                    _, a, b = undo
                    self._apply_block_swap_2x2(board, a, b)
            
            # Cool down
            T *= cfg.sa_alpha
            
            if iteration % 1000 == 0:
                print(f"    Iter {iteration}: T={T:.6f}, current={current_score:.4f}, best={best_score:.4f}")
        
        print(f"  SA best score: {best_score:.4f}")
        return best_board

    def random_pair_hillclimb(self, board: Dict[Tuple[int, int], int], max_iters: int = 120000) -> Dict[Tuple[int, int], int]:
        """Fast approximation of the direct solver's swap refinement using delta scoring."""
        board = dict(board)
        total_edges = 2 * self.grid_size * (self.grid_size - 1)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(total_edges, 1)
        best_score = current_score
        best_board = dict(board)

        positions = list(board.keys())
        for it in range(max_iters):
            a, b = random.sample(positions, 2)
            delta_sum = self._swap_delta_sum(board, a, b)
            if delta_sum < -1e-12:
                board[a], board[b] = board[b], board[a]
                current_sum += delta_sum
                current_score = current_sum / max(total_edges, 1)
                if current_score < best_score:
                    best_score = current_score
                    best_board = dict(board)

            if it % 20000 == 0:
                print(f"    Hillclimb {it}: current={current_score:.4f}, best={best_score:.4f}")

        return best_board
    
    def kl_style_refinement(self, board: Dict) -> Dict:
        """Kernighan-Lin style: swap neighbors that improve adjacency."""
        board = dict(board)
        current_score = self.evaluate(board)
        gs = self.grid_size
        
        print(f"  KL-style refinement (initial: {current_score:.4f})...")
        
        improved = True
        passes = 0
        
        while improved and passes < 5:
            improved = False
            passes += 1
            
            for row in range(gs):
                for col in range(gs):
                    pos = (row, col)
                    pid = board[pos]
                    
                    # Try swapping with each neighbor
                    neighbors = []
                    if row > 0: neighbors.append((row - 1, col))
                    if row < gs - 1: neighbors.append((row + 1, col))
                    if col > 0: neighbors.append((row, col - 1))
                    if col < gs - 1: neighbors.append((row, col + 1))
                    
                    for n_pos in neighbors:
                        n_pid = board[n_pos]
                        
                        # Swap
                        board[pos], board[n_pos] = board[n_pos], board[pos]
                        new_score = self.evaluate(board)
                        
                        if new_score < current_score - 1e-9:
                            current_score = new_score
                            improved = True
                        else:
                            # Undo
                            board[pos], board[n_pos] = board[n_pos], board[pos]
            
            print(f"    Pass {passes}: score={current_score:.4f}")
        
        return board
    
    def refine(self, board: Dict) -> Dict:
        """Full refinement pipeline."""
        print("\n  Starting refinement pipeline...")

        # Phase 1: Quick local improvement
        board = self.local_swap_refinement(board, passes=self.config.refinement_passes, samples_per_pos=64)
        score = self.evaluate(board)

        # Phase 2: Hillclimb swaps (direct-solver style)
        board = self.random_pair_hillclimb(board, max_iters=120000)
        score = self.evaluate(board)

        # Phase 3: SA as a last resort (kept time-bounded)
        if score > self.config.acceptable_threshold:
            board = self.simulated_annealing(board)

        final_score = self.evaluate(board)
        print(f"  Final score after refinement: {final_score:.4f}")
        return board


# =============================================================================
# PHASE 5: AMBIGUITY CLUSTER PERMUTATION SEARCH
# =============================================================================

class AmbiguityClusterRefiner:
    def _final_column_shift_correction(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all cyclic column shifts and pick the one with the lowest score.
        """
        gs = self.grid_size
        best_board = board
        best_score = self.evaluate(board)
        for shift in range(1, gs):
            shifted = {}
            for r in range(gs):
                for c in range(gs):
                    shifted[(r, (c + shift) % gs)] = board[(r, c)]
            score = self.evaluate(shifted)
            if score < best_score - 1e-9:
                best_score = score
                best_board = shifted
        return best_board

    def _final_global_swap_refinement(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all possible pairwise swaps (not just adjacent) and accept any that improve the score.
        Repeat until no improvement.
        """
        gs = self.grid_size
        positions = [(r, c) for r in range(gs) for c in range(gs)]
        improved = True
        best_board = dict(board)
        best_score = self.evaluate(best_board)
        while improved:
            improved = False
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    a, b = positions[i], positions[j]
                    swap_board = dict(best_board)
                    swap_board[a], swap_board[b] = swap_board[b], swap_board[a]
                    score = self.evaluate(swap_board)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_board = swap_board
                        improved = True
        return best_board
    """
    Final deterministic refinement phase that fixes global misplacements.
    
    Identifies clusters of pieces with ambiguous edge matches, then exhaustively
    searches permutations of cluster placements to find the globally optimal
    configuration. This fixes systematic errors like row/column shifts.
    
    Based on techniques from Gallagher 2012 and Paikin & Tal 2015.
    """
    
    def __init__(self, compat: CompatibilityMatrix, grid_size: int, config: SolverConfig):
        self.compat = compat
        self.grid_size = grid_size
        self.config = config
        # Threshold for edge ambiguity (margin between best and second-best)
        self.ambiguity_threshold = 0.03  # Edges with margin < this are ambiguous
        self.max_cluster_size = 16  # Don't permute clusters larger than this
        self.max_total_permutations = 100000  # Global limit on search space
    
    def evaluate(self, board: Dict[Tuple[int, int], int]) -> float:
        """Compute board score (lower = better)."""
        total = 0.0
        count = 0
        gs = self.grid_size
        
        for r in range(gs):
            for c in range(gs):
                pid = board.get((r, c))
                if pid is None:
                    continue
                if c < gs - 1:
                    right = board.get((r, c + 1))
                    if right is not None:
                        total += self.compat.get_horizontal_score(pid, right)
                        count += 1
                if r < gs - 1:
                    bottom = board.get((r + 1, c))
                    if bottom is not None:
                        total += self.compat.get_vertical_score(pid, bottom)
                        count += 1
        
        return total / max(count, 1)
    
    def _get_edge_margins(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], float]:
        """
        For each position, compute the minimum margin of its neighbor edges.
        Lower margin = more ambiguous placement.
        """
        gs = self.grid_size
        margins = {}
        
        for r in range(gs):
            for c in range(gs):
                pid = board[(r, c)]
                pos_margins = []
                
                # Check all 4 neighbors
                if c > 0:  # Left
                    left_pid = board[(r, c - 1)]
                    best = self.compat.best_match.get((left_pid, 'right'))
                    second = self.compat.second_best.get((left_pid, 'right'))
                    if best and second and second[0] is not None:
                        margin = second[2] - best[2]
                        pos_margins.append(margin)
                
                if c < gs - 1:  # Right
                    right_pid = board[(r, c + 1)]
                    best = self.compat.best_match.get((pid, 'right'))
                    second = self.compat.second_best.get((pid, 'right'))
                    if best and second and second[0] is not None:
                        margin = second[2] - best[2]
                        pos_margins.append(margin)
                
                if r > 0:  # Top
                    top_pid = board[(r - 1, c)]
                    best = self.compat.best_match.get((top_pid, 'bottom'))
                    second = self.compat.second_best.get((top_pid, 'bottom'))
                    if best and second and second[0] is not None:
                        margin = second[2] - best[2]
                        pos_margins.append(margin)
                
                if r < gs - 1:  # Bottom
                    bottom_pid = board[(r + 1, c)]
                    best = self.compat.best_match.get((pid, 'bottom'))
                    second = self.compat.second_best.get((pid, 'bottom'))
                    if best and second and second[0] is not None:
                        margin = second[2] - best[2]
                        pos_margins.append(margin)
                
                margins[(r, c)] = min(pos_margins) if pos_margins else float('inf')
        
        return margins
    
    def _find_ambiguous_clusters(self, board: Dict[Tuple[int, int], int]) -> List[Set[Tuple[int, int]]]:
        """
        Find connected components of positions with ambiguous edge matches.
        """
        gs = self.grid_size
        margins = self._get_edge_margins(board)
        
        # Mark ambiguous positions
        ambiguous = set()
        for pos, margin in margins.items():
            if margin < self.ambiguity_threshold:
                ambiguous.add(pos)
        
        if not ambiguous:
            return []
        
        # Find connected components using BFS
        clusters = []
        visited = set()
        
        for start in ambiguous:
            if start in visited:
                continue
            
            cluster = set()
            queue = [start]
            
            while queue:
                pos = queue.pop(0)
                if pos in visited:
                    continue
                visited.add(pos)
                cluster.add(pos)
                
                r, c = pos
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for n in neighbors:
                    if n in ambiguous and n not in visited:
                        queue.append(n)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _get_cluster_shape(self, cluster: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get bounding box of cluster: (min_r, max_r, min_c, max_c)."""
        rows = [p[0] for p in cluster]
        cols = [p[1] for p in cluster]
        return min(rows), max(rows), min(cols), max(cols)
    
    def _try_row_shifts(self, board: Dict[Tuple[int, int], int], 
                        cluster: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """
        Try shifting the cluster to different row positions.
        Returns the best configuration.
        """
        gs = self.grid_size
        min_r, max_r, min_c, max_c = self._get_cluster_shape(cluster)
        cluster_height = max_r - min_r + 1
        cluster_width = max_c - min_c + 1
        
        # Extract cluster pieces
        cluster_pieces = {pos: board[pos] for pos in cluster}
        frozen_pieces = {pos: board[pos] for pos in board if pos not in cluster}
        frozen_pids = set(frozen_pieces.values())
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Try placing cluster at each valid row offset
        for row_offset in range(-(gs - cluster_height), gs):
            new_min_r = min_r + row_offset
            new_max_r = max_r + row_offset
            
            if new_min_r < 0 or new_max_r >= gs:
                continue
            
            # Build candidate board
            candidate = dict(frozen_pieces)
            valid = True
            
            for (r, c), pid in cluster_pieces.items():
                new_pos = (r + row_offset, c)
                if new_pos in frozen_pieces:
                    valid = False
                    break
                candidate[new_pos] = pid
            
            if not valid or len(candidate) != gs * gs:
                continue
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
        
        return best_board
    
    def _try_column_shifts(self, board: Dict[Tuple[int, int], int],
                           cluster: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """
        Try shifting the cluster to different column positions.
        """
        gs = self.grid_size
        min_r, max_r, min_c, max_c = self._get_cluster_shape(cluster)
        cluster_width = max_c - min_c + 1
        
        cluster_pieces = {pos: board[pos] for pos in cluster}
        frozen_pieces = {pos: board[pos] for pos in board if pos not in cluster}
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        for col_offset in range(-(gs - cluster_width), gs):
            new_min_c = min_c + col_offset
            new_max_c = max_c + col_offset
            
            if new_min_c < 0 or new_max_c >= gs:
                continue
            
            candidate = dict(frozen_pieces)
            valid = True
            
            for (r, c), pid in cluster_pieces.items():
                new_pos = (r, c + col_offset)
                if new_pos in frozen_pieces:
                    valid = False
                    break
                candidate[new_pos] = pid
            
            if not valid or len(candidate) != gs * gs:
                continue
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
        
        return best_board
    
    def _try_full_row_permutations(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all permutations of entire rows.
        This addresses the "off by one row" error pattern.
        """
        from itertools import permutations
        
        gs = self.grid_size
        
        # Extract rows
        rows = []
        for r in range(gs):
            row = tuple(board[(r, c)] for c in range(gs))
            rows.append(row)
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Try all row permutations (8! = 40320 for 8x8)
        perm_count = 0
        for perm in permutations(range(gs)):
            perm_count += 1
            
            # Build candidate
            candidate = {}
            for new_r, orig_r in enumerate(perm):
                for c in range(gs):
                    candidate[(new_r, c)] = rows[orig_r][c]
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
        
        return best_board
    
    def _try_full_column_permutations(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all permutations of entire columns.
        """
        from itertools import permutations
        
        gs = self.grid_size
        
        # Extract columns
        cols = []
        for c in range(gs):
            col = tuple(board[(r, c)] for r in range(gs))
            cols.append(col)
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        for perm in permutations(range(gs)):
            candidate = {}
            for new_c, orig_c in enumerate(perm):
                for r in range(gs):
                    candidate[(r, new_c)] = cols[orig_c][r]
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
        
        return best_board
    
    def _try_block_permutations(self, board: Dict[Tuple[int, int], int],
                                 cluster: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """
        Try all internal permutations within a small cluster.
        Only feasible for clusters up to ~8 pieces.
        """
        from itertools import permutations
        
        if len(cluster) > 8:
            return board  # Too expensive
        
        positions = sorted(cluster)
        pieces = [board[pos] for pos in positions]
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        for perm in permutations(pieces):
            candidate = dict(board)
            for pos, pid in zip(positions, perm):
                candidate[pos] = pid
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
        
        return best_board
    
    def _rebuild_optimal_rows(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try a multi-start beam search with DIVERSITY to find better row configurations.
        The problem is that normal beam search prunes the correct rows too early.
        
        Approach: Build rows from EACH piece as a starting point, then combine.
        """
        from itertools import permutations
        import heapq
        
        gs = self.grid_size
        all_pieces = list(range(64))
        
        print("    Rebuilding with diverse row generation...")
        
        # For each piece, build the best rows starting from that piece
        all_complete_rows = {}  # row_tuple -> score
        
        for start_piece in all_pieces:
            # Build rows starting from this piece with small beam
            row_states = [(tuple([start_piece]), frozenset([start_piece]), 0.0)]
            
            for col in range(1, gs):
                new_states = []
                for row, used, score in row_states:
                    last = row[-1]
                    for pid in all_pieces:
                        if pid in used:
                            continue
                        edge_score = self.compat.get_horizontal_score(last, pid)
                        new_states.append((row + (pid,), used | {pid}, score + edge_score))
                
                # Keep top 500 for this starting piece
                new_states.sort(key=lambda x: x[2] / col)
                row_states = new_states[:500]
            
            # Add to global pool
            for row, used, score in row_states:
                avg_score = score / (gs - 1)
                if row not in all_complete_rows or avg_score < all_complete_rows[row]:
                    all_complete_rows[row] = avg_score
        
        print(f"      Collected {len(all_complete_rows)} unique rows from diverse starts")
        
        # Convert to sorted list
        complete_rows = sorted(all_complete_rows.items(), key=lambda x: x[1])
        
        if len(complete_rows) < 8:
            print("      Not enough rows generated")
            return board
        
        print(f"      Best row score: {complete_rows[0][1]:.4f}, Worst: {complete_rows[-1][1]:.4f}")
        
        # Try greedy selection first
        selected_rows = []
        used_pieces = set()
        
        for row, score in complete_rows:
            row_pieces = set(row)
            if row_pieces & used_pieces:
                continue
            selected_rows.append(row)
            used_pieces |= row_pieces
            if len(selected_rows) == gs:
                break
        
        if len(selected_rows) < gs:
            print(f"      Greedy got {len(selected_rows)} rows, trying DFS...")
            
            # DFS with larger pool
            max_pool = min(len(complete_rows), 20000)
            pool = complete_rows[:max_pool]
            
            row_entries = []
            for row, score in pool:
                mask = 0
                for pid in row:
                    mask |= (1 << int(pid))
                row_entries.append((mask, row, score))
            
            best_rows = None
            best_cost = float('inf')
            nodes = [0]
            
            def dfs(chosen, used_mask, cost):
                nonlocal best_rows, best_cost
                nodes[0] += 1
                if nodes[0] > 1000000:
                    return
                
                if len(chosen) == gs:
                    if cost < best_cost:
                        best_cost = cost
                        best_rows = [row_entries[i][1] for i in chosen]
                    return
                
                if cost >= best_cost:
                    return
                
                # Find uncovered piece
                for pid in all_pieces:
                    if not (used_mask & (1 << pid)):
                        pivot = pid
                        break
                else:
                    return
                
                # Find rows containing pivot
                options = [(i, row_entries[i][2]) for i, (m, r, s) in enumerate(row_entries) 
                           if pivot in r and not (m & used_mask)]
                options.sort(key=lambda x: x[1])
                
                for ridx, _ in options[:200]:
                    mask = row_entries[ridx][0]
                    score = row_entries[ridx][2]
                    chosen.append(ridx)
                    dfs(chosen, used_mask | mask, cost + score)
                    chosen.pop()
                    if nodes[0] > 1000000:
                        return
            
            dfs([], 0, 0.0)
            print(f"      DFS explored {nodes[0]} nodes")
            
            if best_rows:
                selected_rows = best_rows
                print(f"      Found {len(selected_rows)} disjoint rows")
            else:
                print("      Failed to find 8 disjoint rows")
                return board
        
        # Optimal row ordering
        def row_vertical_cost(row_top, row_bottom):
            return sum(self.compat.get_vertical_score(row_top[c], row_bottom[c]) for c in range(gs)) / gs
        
        best_order = None
        best_total = float('inf')
        
        for perm in permutations(range(gs)):
            cost = sum(row_vertical_cost(selected_rows[perm[i]], selected_rows[perm[i+1]]) for i in range(gs-1))
            if cost < best_total:
                best_total = cost
                best_order = perm
        
        new_board = {}
        for row_idx, orig_idx in enumerate(best_order):
            for col_idx, pid in enumerate(selected_rows[orig_idx]):
                new_board[(row_idx, col_idx)] = pid
        
        new_score = self.evaluate(new_board)
        old_score = self.evaluate(board)
        
        print(f"      Rebuilt score: {new_score:.4f} (was {old_score:.4f})")
        
        return new_board if new_score < old_score - 1e-9 else board

    def _try_2d_global_construction(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Build the puzzle from scratch using 2D scoring (H + V simultaneously).
        Instead of building rows first, build the entire grid considering both edges.
        
        Uses a greedy spiral fill from the center outward.
        """
        gs = self.grid_size
        all_pieces = list(set(board.values()))
        
        print("    Trying 2D global construction...")
        
        # Start from multiple seeds and take the best
        best_board = board
        best_score = self.evaluate(board)
        
        # Focus on corners and edges (where pieces have fewer constraints)
        # These are more reliable starting points
        start_positions = [
            (0, 0), (0, gs-1), (gs-1, 0), (gs-1, gs-1),  # corners
        ]
        
        trials = 0
        # For corners, try all 64 pieces as seeds
        for start_row, start_col in start_positions:
            for seed_piece in all_pieces:
                new_board = self._build_2d_from_seed(seed_piece, start_row, start_col, all_pieces)
                trials += 1
                
                if new_board:
                    new_score = self.evaluate(new_board)
                    if new_score < best_score - 1e-6:
                        best_score = new_score
                        best_board = new_board
        
        print(f"      Tried {trials} 2D constructions")
        if best_score < self.evaluate(board) - 1e-6:
            print(f"      2D construction improved: {self.evaluate(board):.4f} -> {best_score:.4f}")
        else:
            print(f"      2D construction: no improvement")
        
        return best_board
    
    def _build_2d_from_seed(self, seed: int, start_r: int, start_c: int, 
                            all_pieces: List[int]) -> Optional[Dict[Tuple[int, int], int]]:
        """Build puzzle greedily from a seed piece using combined H+V scoring."""
        gs = self.grid_size
        board = {(start_r, start_c): seed}
        used = {seed}
        
        # BFS order: positions to fill next
        from collections import deque
        queue = deque()
        
        # Add adjacent positions to queue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < gs and 0 <= nc < gs:
                queue.append((nr, nc))
        
        visited = {(start_r, start_c)}
        for item in queue:
            visited.add(item)
        
        while queue and len(board) < 64:
            r, c = queue.popleft()
            
            # Find best piece for this position considering ALL neighbors
            best_piece = None
            best_cost = float('inf')
            
            for pid in all_pieces:
                if pid in used:
                    continue
                
                cost = 0.0
                count = 0
                
                # Check left neighbor
                if c > 0 and (r, c-1) in board:
                    left = board[(r, c-1)]
                    cost += self.compat.get_horizontal_score(left, pid)
                    count += 1
                
                # Check right neighbor
                if c < gs-1 and (r, c+1) in board:
                    right = board[(r, c+1)]
                    cost += self.compat.get_horizontal_score(pid, right)
                    count += 1
                
                # Check top neighbor
                if r > 0 and (r-1, c) in board:
                    top = board[(r-1, c)]
                    cost += self.compat.get_vertical_score(top, pid)
                    count += 1
                
                # Check bottom neighbor
                if r < gs-1 and (r+1, c) in board:
                    bottom = board[(r+1, c)]
                    cost += self.compat.get_vertical_score(pid, bottom)
                    count += 1
                
                if count > 0:
                    avg_cost = cost / count
                    if avg_cost < best_cost:
                        best_cost = avg_cost
                        best_piece = pid
            
            if best_piece is None:
                return None  # Failed to fill
            
            board[(r, c)] = best_piece
            used.add(best_piece)
            
            # Add new adjacent positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs and (nr, nc) not in visited:
                    queue.append((nr, nc))
                    visited.add((nr, nc))
        
        return board if len(board) == 64 else None

    def _try_k_piece_swaps(self, board: Dict[Tuple[int, int], int], k: int = 4) -> Dict[Tuple[int, int], int]:
        """
        Try all permutations of k pieces that might be misplaced.
        Focus on pieces with worst local scores.
        """
        from itertools import permutations, combinations
        
        gs = self.grid_size
        
        # Find pieces with worst local scores
        piece_scores = []
        for r in range(gs):
            for c in range(gs):
                pid = board[(r, c)]
                local_score = 0.0
                count = 0
                
                if c > 0:
                    local_score += self.compat.get_horizontal_score(board[(r, c-1)], pid)
                    count += 1
                if c < gs - 1:
                    local_score += self.compat.get_horizontal_score(pid, board[(r, c+1)])
                    count += 1
                if r > 0:
                    local_score += self.compat.get_vertical_score(board[(r-1, c)], pid)
                    count += 1
                if r < gs - 1:
                    local_score += self.compat.get_vertical_score(pid, board[(r+1, c)])
                    count += 1
                
                if count > 0:
                    piece_scores.append(((r, c), local_score / count))
        
        # Sort by worst score (highest = worst)
        piece_scores.sort(key=lambda x: -x[1])
        
        # Take top 12 worst-fitting pieces
        candidates = [pos for pos, _ in piece_scores[:12]]
        
        best_board = board
        best_score = self.evaluate(board)
        
        # Try all k-subsets of these candidates
        for subset in combinations(candidates, k):
            positions = list(subset)
            pieces = [board[pos] for pos in positions]
            
            # Try all permutations of these pieces
            for perm in permutations(pieces):
                if perm == tuple(pieces):
                    continue
                
                # Apply permutation
                test_board = dict(board)
                for i, pos in enumerate(positions):
                    test_board[pos] = perm[i]
                
                score = self.evaluate(test_board)
                if score < best_score - 1e-6:
                    best_score = score
                    best_board = test_board
        
        return best_board

    def refine(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Main refinement entry point.
        """
        print("\n  Ambiguity cluster permutation search...")
        
        initial_score = self.evaluate(board)
        print(f"    Initial score: {initial_score:.4f}")
        
        # Step 0a: Try 2D global construction (uses both H and V from start)
        board = self._try_2d_global_construction(board)
        
        # Step 0b: Try rebuilding optimal rows from scratch
        board = self._rebuild_optimal_rows(board)
        rebuild_score = self.evaluate(board)
        
        # Step 1: Find ambiguous clusters
        clusters = self._find_ambiguous_clusters(board)
        print(f"    Found {len(clusters)} ambiguous clusters")
        
        if clusters:
            sizes = sorted([len(c) for c in clusters], reverse=True)
            print(f"    Cluster sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''}")
        
        # Step 2: Try row permutations (addresses row-shift errors)
        print("    Trying row permutations...")
        board = self._try_full_row_permutations(board)
        row_score = self.evaluate(board)
        if row_score < initial_score - 1e-6:
            print(f"      Improved by row permutation: {initial_score:.4f} -> {row_score:.4f}")
        
        # Step 3: Try column permutations
        print("    Trying column permutations...")
        board = self._try_full_column_permutations(board)
        col_score = self.evaluate(board)
        if col_score < row_score - 1e-6:
            print(f"      Improved by column permutation: {row_score:.4f} -> {col_score:.4f}")
        
        # Step 4: Process individual clusters
        for i, cluster in enumerate(clusters):
            if len(cluster) > self.max_cluster_size:
                print(f"    Skipping cluster {i+1} (size {len(cluster)} > {self.max_cluster_size})")
                continue
            
            # Try row shifts for this cluster
            before = self.evaluate(board)
            board = self._try_row_shifts(board, cluster)
            after = self.evaluate(board)
            if after < before - 1e-6:
                print(f"    Cluster {i+1} row shift: {before:.4f} -> {after:.4f}")
            
            # Try column shifts
            before = self.evaluate(board)
            board = self._try_column_shifts(board, cluster)
            after = self.evaluate(board)
            if after < before - 1e-6:
                print(f"    Cluster {i+1} col shift: {before:.4f} -> {after:.4f}")
            
            # Try internal permutations for small clusters
            if len(cluster) <= 8:
                before = self.evaluate(board)
                board = self._try_block_permutations(board, cluster)
                after = self.evaluate(board)
                if after < before - 1e-6:
                    print(f"    Cluster {i+1} internal perm: {before:.4f} -> {after:.4f}")
        
        # Step 5: Final row/column permutations after cluster refinement
        print("    Final row/column sweep...")
        board = self._try_full_row_permutations(board)
        board = self._try_full_column_permutations(board)
        
        # Step 6/7: Alternate global swap and column shift correction until convergence
        print("    Alternating global swap and column shift correction...")
        prev_score = self.evaluate(board)
        for _ in range(10):  # Limit to 10 cycles to avoid infinite loop
            before = self.evaluate(board)
            board = self._final_global_swap_refinement(board)
            after = self.evaluate(board)
            if after < before - 1e-6:
                print(f"      Global swap improved: {before:.4f} -> {after:.4f}")
            before = after
            board = self._final_column_shift_correction(board)
            after = self.evaluate(board)
            if after < before - 1e-6:
                print(f"      Column shift improved: {before:.4f} -> {after:.4f}")
            if abs(after - prev_score) < 1e-9:
                break
            prev_score = after

        final_score = self.evaluate(board)
        print(f"    Final score after cluster refinement: {final_score:.4f}")

        if final_score < initial_score - 1e-6:
            print(f"    IMPROVED: {initial_score:.4f} -> {final_score:.4f}")
        else:
            print(f"    No improvement found")

        return board


# =============================================================================
# PHASE 4: DIAGNOSTICS & VISUALIZATION
# =============================================================================

class PuzzleDiagnostics:
    """Diagnostics and visualization."""
    
    def __init__(self, pieces: Dict[int, np.ndarray], compat: CompatibilityMatrix,
                 detector: ConfidentPairDetector, config: SolverConfig):
        self.pieces = pieces
        self.compat = compat
        self.detector = detector
        self.config = config
        self.debug_dir = Path(config.debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def save_locked_pairs_matrix(self):
        """Save locked pairs as a matrix heatmap."""
        n = len(self.compat.piece_ids)
        matrix = np.zeros((n, n))
        
        id_to_idx = {pid: i for i, pid in enumerate(sorted(self.compat.piece_ids))}
        
        for pair in self.detector.locked_pairs:
            i, j = id_to_idx[pair.piece_a], id_to_idx[pair.piece_b]
            matrix[i, j] = 1
            matrix[j, i] = 1
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(matrix, cmap='Blues')
        ax.set_title(f'Locked Pairs ({len(self.detector.locked_pairs)} pairs)')
        ax.set_xlabel('Piece ID')
        ax.set_ylabel('Piece ID')
        
        plt.tight_layout()
        path = self.debug_dir / "locked_pairs_matrix.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  Saved: {path}")
    
    def save_compatibility_heatmap(self, edge_type: str = 'horizontal'):
        """Save compatibility heatmap."""
        n = len(self.compat.piece_ids)
        pids = sorted(self.compat.piece_ids)
        matrix = np.zeros((n, n))
        
        for i, pid_a in enumerate(pids):
            for j, pid_b in enumerate(pids):
                if pid_a == pid_b:
                    matrix[i, j] = np.nan
                else:
                    if edge_type == 'horizontal':
                        matrix[i, j] = self.compat.get_horizontal_score(pid_a, pid_b)
                    else:
                        matrix[i, j] = self.compat.get_vertical_score(pid_a, pid_b)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='viridis_r')
        plt.colorbar(im, ax=ax, label='Score (lower = better)')
        
        ax.set_title(f'Compatibility: {edge_type}')
        ax.set_xlabel('Piece B')
        ax.set_ylabel('Piece A')
        
        plt.tight_layout()
        path = self.debug_dir / f"compatibility_{edge_type}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  Saved: {path}")
    
    def save_confidence_histogram(self):
        """Save confidence ratio histogram."""
        ratios = [p.confidence_ratio for p in self.detector.locked_pairs]
        margins = [p.margin for p in self.detector.locked_pairs]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(ratios, bins=30, edgecolor='black')
        axes[0].set_xlabel('Confidence Ratio')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Locked Pairs: Confidence Ratio Distribution')
        
        axes[1].hist(margins, bins=30, edgecolor='black')
        axes[1].set_xlabel('Margin')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Locked Pairs: Margin Distribution')
        
        plt.tight_layout()
        path = self.debug_dir / "confidence_histogram.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  Saved: {path}")
    
    def save_solution_image(self, board: Dict[Tuple[int, int], int], 
                            score: float, filename: str = "solution.png"):
        """Save assembled puzzle image."""
        gs = int(np.sqrt(len(board)))
        first = self.pieces[0]
        h, w = first.shape[:2]
        is_color = len(first.shape) == 3
        
        if is_color:
            assembled = np.zeros((h * gs, w * gs, 3), dtype=np.uint8)
        else:
            assembled = np.zeros((h * gs, w * gs), dtype=np.uint8)
        
        for (row, col), pid in board.items():
            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w
            assembled[y1:y2, x1:x2] = self.pieces[pid]
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        if is_color:
            ax.imshow(cv2.cvtColor(assembled, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(assembled, cmap='gray')
        
        ax.set_title(f'Solution (Score: {score:.4f})', fontsize=14, fontweight='bold')
        
        # Add piece labels
        for (row, col), pid in board.items():
            x_pos = col * w + w // 4
            y_pos = row * h + h // 4
            ax.text(x_pos, y_pos, f'{pid}', color='red', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        path = self.debug_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    
    def run_all_diagnostics(self, board: Dict, score: float):
        """Run all diagnostics."""
        print("\n  Generating diagnostics...")
        self.save_locked_pairs_matrix()
        self.save_compatibility_heatmap('horizontal')
        self.save_compatibility_heatmap('vertical')
        self.save_confidence_histogram()
        self.save_solution_image(board, score)


# =============================================================================
# MAIN SOLVER
# =============================================================================

def load_puzzle(puzzle_path: str) -> Tuple[Dict[int, np.ndarray], int]:
    """Load puzzle pieces."""
    img = cv2.imread(puzzle_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load: {puzzle_path}")
    
    h, w = img.shape[:2]
    
    for gs in [8, 4, 2]:
        if h % gs == 0 and w % gs == 0:
            break
    
    piece_h, piece_w = h // gs, w // gs
    
    pieces = {}
    idx = 0
    for row in range(gs):
        for col in range(gs):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            pieces[idx] = img[y1:y2, x1:x2].copy()
            idx += 1
    
    return pieces, gs


def solve_puzzle(puzzle_path: str, config: Optional[SolverConfig] = None) -> Dict:
    """
    Main solving function with full pipeline.
    """
    if config is None:
        config = SolverConfig()
    
    print("=" * 70)
    print("IMPROVED 8x8 PUZZLE SOLVER")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load
    print("\n[Phase 0] Loading and computing descriptors...")
    pieces, grid_size = load_puzzle(puzzle_path)
    print(f"  Loaded {len(pieces)} pieces ({grid_size}x{grid_size})")
    
    # Compatibility matrix
    compat = CompatibilityMatrix(pieces, config)
    
    # Phase 1: Confident pairs
    print("\n[Phase 1] Detecting confident pairs...")
    detector = ConfidentPairDetector(compat, config)
    detector.detect_pairs()
    detector.build_superpieces()
    
    # Phase 2: Assembly
    print("\n[Phase 2] Assembling puzzle...")
    assembler = PuzzleAssembler(compat, detector, grid_size, config)
    board = assembler.assemble()
    assembly_score = assembler.evaluate_board(board)
    print(f"  Assembly score: {assembly_score:.4f}")
    
    # Phase 3: Refinement
    print("\n[Phase 3] Refining solution...")
    refiner = PuzzleRefiner(compat, grid_size, config)
    board = refiner.refine(board)
    phase3_score = refiner.evaluate(board)
    
    # Phase 5: Ambiguity Cluster Permutation Search
    print("\n[Phase 5] Ambiguity cluster permutation search...")
    cluster_refiner = AmbiguityClusterRefiner(compat, grid_size, config)
    board = cluster_refiner.refine(board)
    final_score = cluster_refiner.evaluate(board)
    
    # Phase 4: Diagnostics
    print("\n[Phase 4] Diagnostics...")
    diagnostics = PuzzleDiagnostics(pieces, compat, detector, config)
    diagnostics.run_all_diagnostics(board, final_score)
    
    # Final summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SOLUTION COMPLETE")
    print("=" * 70)
    print(f"Final Score: {final_score:.4f}")
    print(f"Locked Pairs: {len(detector.locked_pairs)}")
    print(f"Superpieces: {len(detector.superpieces)}")
    print(f"Time: {elapsed:.1f}s")
    
    # Arrangement
    arrangement = []
    for row in range(grid_size):
        for col in range(grid_size):
            arrangement.append(board[(row, col)])
    print(f"Arrangement: {arrangement}")
    
    return {
        'board': board,
        'arrangement': tuple(arrangement),
        'score': final_score,
        'grid_size': grid_size,
        'locked_pairs': len(detector.locked_pairs),
        'superpieces': len(detector.superpieces),
        'time': elapsed
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        puzzle_path = sys.argv[1]
    else:
        puzzle_path = "./Gravity Falls/puzzle_8x8/0.jpg"
    
    config = SolverConfig(
        debug_dir="./debug",
        verbose=True,
        sa_max_iters=3000,
        sa_time_limit=20.0
    )
    
    result = solve_puzzle(puzzle_path, config)
