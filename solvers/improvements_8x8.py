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
    sa_max_iters: int = 3000
    sa_time_limit: float = 20.0  # seconds
    
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

def border_likelihood(edge_strip: np.ndarray) -> float:
    """
    Compute how likely an edge is to be a TRUE puzzle border (not internal edge).
    True borders have LOW gradient variance (smooth/uniform outside the puzzle).
    Returns variance of gradients - LOWER = more likely to be true border.
    """
    if edge_strip is None or edge_strip.size == 0:
        return 1.0  # High value = not a border
    
    # Convert to grayscale if needed
    if len(edge_strip.shape) == 3:
        gray = cv2.cvtColor(edge_strip, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_strip.copy()
    
    # Compute gradients
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.abs(gx) + np.abs(gy)
    
    # Return variance of gradient magnitude
    # Low variance = uniform edge = likely true border
    return float(np.var(g))


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
        
        # Compute border likelihood (how likely this edge is a TRUE puzzle border)
        # Uses a 5-pixel strip at the edge boundary
        border_strip = self._get_border_strip(piece, edge, width=5)
        self.border_likelihood = border_likelihood(border_strip)
    
    def _get_border_strip(self, piece: np.ndarray, edge: str, width: int = 5) -> np.ndarray:
        """Get a strip at the very edge of the piece for border detection."""
        h, w = piece.shape[:2]
        width = min(width, h, w)  # Don't exceed piece dimensions
        
        if edge == 'top':
            return piece[:width, :, :] if len(piece.shape) == 3 else piece[:width, :]
        elif edge == 'bottom':
            return piece[-width:, :, :] if len(piece.shape) == 3 else piece[-width:, :]
        elif edge == 'left':
            return piece[:, :width, :] if len(piece.shape) == 3 else piece[:, :width]
        elif edge == 'right':
            return piece[:, -width:, :] if len(piece.shape) == 3 else piece[:, -width:]
        return None
    
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
    
    # Border penalty weight - higher = stronger enforcement of true borders
    BORDER_PENALTY_WEIGHT = 0.005
    
    def __init__(self, pieces: Dict[int, np.ndarray], config: SolverConfig):
        self.config = config
        self.piece_ids = sorted(pieces.keys())
        self.n_pieces = len(self.piece_ids)
        self.grid_size = int(np.sqrt(self.n_pieces))
        self.edges = ['top', 'bottom', 'left', 'right']
        self.complementary = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        
        # Compute all descriptors
        print("  Computing edge descriptors...")
        self.descriptors: Dict[Tuple[int, str], EdgeDescriptors] = {}
        for pid in self.piece_ids:
            for edge in self.edges:
                self.descriptors[(pid, edge)] = EdgeDescriptors(pieces[pid], edge, config)
        
        # Normalize border likelihoods to [0, 1] range
        self._normalize_border_likelihoods()
        
        # Compute all compatibilities
        print("  Computing pairwise compatibility...")
        # compatibility[(pid_a, edge_a, pid_b, edge_b)] = score
        self.compatibility: Dict[Tuple[int, str, int, str], float] = {}
        
        # best_match[(pid, edge)] = (best_pid, best_edge, best_score)
        self.best_match: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        
        # second_best_match[(pid, edge)] = (pid, edge, score)
        self.second_best: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        
    def _normalize_border_likelihoods(self):
        """Normalize border likelihood values to [0, 1] range."""
        all_likelihoods = [self.descriptors[(pid, edge)].border_likelihood 
                          for pid in self.piece_ids for edge in self.edges]
        
        min_val = min(all_likelihoods)
        max_val = max(all_likelihoods)
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        # Normalize: low original value -> low normalized value -> more likely to be border
        for pid in self.piece_ids:
            for edge in self.edges:
                desc = self.descriptors[(pid, edge)]
                desc.border_likelihood_normalized = (desc.border_likelihood - min_val) / range_val
        
        print(f"  Border likelihood range: [{min_val:.2f}, {max_val:.2f}]")
    
    def get_border_penalty(self, pid: int, row: int, col: int) -> float:
        """
        Compute border penalty for placing piece pid at position (row, col).
        Pieces on the border should have LOW border_likelihood (smooth edges).
        If a piece with HIGH border_likelihood is placed on border, add penalty.
        If a piece with LOW border_likelihood is NOT on border, add penalty.
        """
        gs = self.grid_size
        penalty = 0.0
        weight = self.BORDER_PENALTY_WEIGHT
        
        # Check each edge
        # TOP edge of puzzle (row == 0): piece's top edge should be border-like (low likelihood)
        if row == 0:
            top_likelihood = self.descriptors[(pid, 'top')].border_likelihood_normalized
            penalty += weight * top_likelihood  # Penalize high likelihood on border
        else:
            # NOT on top border - penalize if this edge looks like a border (should be internal)
            top_likelihood = self.descriptors[(pid, 'top')].border_likelihood_normalized
            penalty += weight * (1.0 - top_likelihood) * 0.5  # Mild penalty for border-like internal edges
        
        # BOTTOM edge of puzzle (row == gs-1)
        if row == gs - 1:
            bottom_likelihood = self.descriptors[(pid, 'bottom')].border_likelihood_normalized
            penalty += weight * bottom_likelihood
        else:
            bottom_likelihood = self.descriptors[(pid, 'bottom')].border_likelihood_normalized
            penalty += weight * (1.0 - bottom_likelihood) * 0.5
        
        # LEFT edge of puzzle (col == 0)
        if col == 0:
            left_likelihood = self.descriptors[(pid, 'left')].border_likelihood_normalized
            penalty += weight * left_likelihood
        else:
            left_likelihood = self.descriptors[(pid, 'left')].border_likelihood_normalized
            penalty += weight * (1.0 - left_likelihood) * 0.5
        
        # RIGHT edge of puzzle (col == gs-1)
        if col == gs - 1:
            right_likelihood = self.descriptors[(pid, 'right')].border_likelihood_normalized
            penalty += weight * right_likelihood
        else:
            right_likelihood = self.descriptors[(pid, 'right')].border_likelihood_normalized
            penalty += weight * (1.0 - right_likelihood) * 0.5
        
        return penalty
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
        """Compute average edge score + border penalties for current board."""
        if board is None:
            board = self.board
        
        total = 0.0
        count = 0
        border_penalty_total = 0.0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in board:
                    continue
                pid = board[(row, col)]
                
                # Add border penalty for this piece at this position
                border_penalty_total += self.compat.get_border_penalty(pid, row, col)
                
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
        
        seam_score = total / max(count, 1)
        return seam_score + border_penalty_total
    
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
        
        # === DFS SAFETY GUARDS ===
        DFS_TIME_LIMIT = 12.0        # Wall-clock seconds
        DFS_NODE_LIMIT = 400_000     # Max recursive calls
        PARTIAL_MIN_ROWS = 4         # Min rows to accept partial solution
        
        dfs_start_time = time.time()
        dfs_nodes = [0]
        dfs_aborted = [False]
        abort_reason = [""]
        
        # Track best partial solution
        best_partial_rows: List[Optional[List[Tuple[int, ...]]]] = [None]
        best_partial_depth = [0]
        best_partial_cost = [float('inf')]

        def dfs(chosen: List[int], used_mask: int, cumulative_cost: float) -> bool:
            nonlocal best_rows
            dfs_nodes[0] += 1
            
            # === GUARD 1: Node limit ===
            if dfs_nodes[0] >= DFS_NODE_LIMIT:
                dfs_aborted[0] = True
                abort_reason[0] = f"node limit ({DFS_NODE_LIMIT})"
                return False
            
            # === GUARD 2: Time limit (check periodically) ===
            if dfs_nodes[0] % 2000 == 0:
                if time.time() - dfs_start_time > DFS_TIME_LIMIT:
                    dfs_aborted[0] = True
                    abort_reason[0] = f"time limit ({DFS_TIME_LIMIT}s)"
                    return False
            
            # === Track best partial solution ===
            depth = len(chosen)
            if depth > best_partial_depth[0] or (depth == best_partial_depth[0] and cumulative_cost < best_partial_cost[0]):
                best_partial_depth[0] = depth
                best_partial_cost[0] = cumulative_cost
                best_partial_rows[0] = [row_entries[i][1] for i in chosen]
            
            # === Complete solution found ===
            if depth == gs:
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
                if dfs_aborted[0]:
                    return False
                mask, _, row_cost = row_entries[ridx]
                if mask & used_mask:
                    continue
                chosen.append(ridx)
                if dfs(chosen, used_mask | mask, cumulative_cost + row_cost):
                    return True
                chosen.pop()
            return False

        found = dfs([], 0, 0.0)
        
        # === Report DFS result ===
        elapsed = time.time() - dfs_start_time
        if dfs_aborted[0]:
            print(f"      DFS aborted: {abort_reason[0]} ({dfs_nodes[0]} nodes)")
            print(f"      Best partial depth: {best_partial_depth[0]} rows")
        else:
            print(f"      DFS explored {dfs_nodes[0]} nodes in {elapsed:.2f}s")
        
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
        """Compute board score including border penalties."""
        total = 0.0
        count = 0
        gs = self.grid_size
        border_penalty_total = 0.0
        
        for row in range(gs):
            for col in range(gs):
                pid = board.get((row, col))
                if pid is None:
                    continue
                
                # Add border penalty for this piece at this position
                border_penalty_total += self.compat.get_border_penalty(pid, row, col)
                
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
        
        seam_score = total / max(count, 1)
        return seam_score + border_penalty_total

    def _total_edge_sum(self, board: Dict[Tuple[int, int], int]) -> float:
        """Sum of all neighbor edge scores + border penalties for a FULL board."""
        total = 0.0
        gs = self.grid_size
        for r in range(gs):
            for c in range(gs):
                pid = board[(r, c)]
                # Add border penalty
                total += self.compat.get_border_penalty(pid, r, c)
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
        
        # Border penalty delta: only positions a and b change their pieces
        ra, ca = a
        rb, cb = b
        pid_a_old = board[a]
        pid_b_old = board[b]
        # After swap: position a has pid_b_old, position b has pid_a_old
        old_sum += self.compat.get_border_penalty(pid_a_old, ra, ca)
        old_sum += self.compat.get_border_penalty(pid_b_old, rb, cb)
        new_sum += self.compat.get_border_penalty(pid_b_old, ra, ca)  # pid_b now at position a
        new_sum += self.compat.get_border_penalty(pid_a_old, rb, cb)  # pid_a now at position b
        
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

    def random_pair_hillclimb(self, board: Dict[Tuple[int, int], int], max_iters: int = 80000) -> Dict[Tuple[int, int], int]:
        """Fast approximation of the direct solver's swap refinement using delta scoring."""
        board = dict(board)
        total_edges = 2 * self.grid_size * (self.grid_size - 1)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(total_edges, 1)
        best_score = current_score
        best_board = dict(board)

        positions = list(board.keys())
        no_improve_count = 0
        
        for it in range(max_iters):
            a, b = random.sample(positions, 2)
            delta_sum = self._swap_delta_sum(board, a, b)
            if delta_sum < -1e-12:
                board[a], board[b] = board[b], board[a]
                current_sum += delta_sum
                current_score = current_sum / max(total_edges, 1)
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_board = dict(board)
            else:
                no_improve_count += 1
            
            # Early termination if no improvement for a while
            if no_improve_count > 15000:
                break

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
        board = self.local_swap_refinement(board, passes=self.config.refinement_passes, samples_per_pos=32)
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

    def _try_all_cyclic_shifts(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all combinations of global cyclic row and column shifts.
        This fixes cases where the entire puzzle is shifted by N columns or M rows.
        """
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Try all column shifts (0 to gs-1)
        for col_shift in range(gs):
            # Try all row shifts (0 to gs-1)
            for row_shift in range(gs):
                if col_shift == 0 and row_shift == 0:
                    continue
                
                shifted = {}
                for r in range(gs):
                    for c in range(gs):
                        new_r = (r + row_shift) % gs
                        new_c = (c + col_shift) % gs
                        shifted[(new_r, new_c)] = board[(r, c)]
                
                score = self.evaluate(shifted)
                if score < best_score - 1e-9:
                    best_score = score
                    best_board = shifted
                    print(f"        Cyclic shift (row={row_shift}, col={col_shift}) improved: {best_score:.4f}")
        
        return best_board

    def _try_row_cyclic_shifts(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try cyclic shifts for individual rows to fix row misalignment.
        This fixes cases where rows are internally correct but shifted relative to each other.
        """
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        improved = True
        while improved:
            improved = False
            for row in range(gs):
                for shift in range(1, gs):
                    candidate = dict(best_board)
                    # Shift this row by 'shift' positions
                    for c in range(gs):
                        candidate[(row, (c + shift) % gs)] = best_board[(row, c)]
                    
                    score = self.evaluate(candidate)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_board = candidate
                        improved = True
                        print(f"        Row {row} shift by {shift} improved: {best_score:.4f}")
        
        return best_board

    def _try_column_block_shifts(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try shifting blocks of columns (e.g., last 3 columns become first 3).
        This addresses the specific issue where column groups are misplaced.
        """
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Try all possible column reorderings by shifting blocks
        for shift in range(1, gs):
            # Shift all columns: column c becomes column (c + shift) % gs
            candidate = {}
            for r in range(gs):
                for c in range(gs):
                    new_c = (c + shift) % gs
                    candidate[(r, new_c)] = board[(r, c)]
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
                print(f"        Column block shift by {shift} improved: {best_score:.4f}")
        
        return best_board

    def _try_all_row_shift_combinations(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all combinations of row shifts to find the globally optimal alignment.
        For 8 rows with 8 possible shifts each, this is 8^8 = 16M combinations, which is too many.
        Instead, we use a greedy approach: fix one row and find best shifts for others.
        """
        from itertools import product
        
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Fix row 0 and try all shifts for other rows
        # For each row, try all 8 shifts and pick the best relative to fixed rows
        for anchor_shift in range(gs):
            candidate = dict(board)
            
            # Shift all rows by anchor_shift first
            for r in range(gs):
                for c in range(gs):
                    candidate[(r, (c + anchor_shift) % gs)] = board[(r, c)]
            
            # Now greedily adjust each row
            for row in range(1, gs):
                row_best_shift = 0
                row_best_score = self.evaluate(candidate)
                
                for shift in range(1, gs):
                    test = dict(candidate)
                    for c in range(gs):
                        test[(row, (c + shift) % gs)] = candidate[(row, c)]
                    
                    score = self.evaluate(test)
                    if score < row_best_score - 1e-9:
                        row_best_score = score
                        row_best_shift = shift
                
                if row_best_shift != 0:
                    new_candidate = dict(candidate)
                    for c in range(gs):
                        new_candidate[(row, (c + row_best_shift) % gs)] = candidate[(row, c)]
                    candidate = new_candidate
            
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score = score
                best_board = candidate
                print(f"        Global row alignment (anchor={anchor_shift}) improved: {best_score:.4f}")
        
        return best_board

    def _try_column_reordering_by_vertical_fit(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try reordering columns to maximize vertical edge fit.
        Build columns one at a time using best vertical compatibility.
        """
        gs = self.grid_size
        
        # Extract columns
        columns = []
        for c in range(gs):
            col = tuple(board[(r, c)] for r in range(gs))
            columns.append(col)
        
        # Calculate vertical cost of putting col_a left of col_b
        def col_vertical_cost(col_a, col_b):
            # This is actually horizontal cost - col_a's right edges match col_b's left edges
            cost = 0.0
            for r in range(gs):
                cost += self.compat.get_horizontal_score(col_a[r], col_b[r])
            return cost / gs
        
        # Find best ordering of columns using beam search
        beam_width = 1000
        beam = [(tuple([0]), frozenset([0]), 0.0)]  # Start with column 0
        
        for _ in range(1, gs):
            new_beam = []
            for order, used, cum_cost in beam:
                for c in range(gs):
                    if c in used:
                        continue
                    # Cost of adding column c after the last column in order
                    add_cost = col_vertical_cost(columns[order[-1]], columns[c])
                    new_beam.append((order + (c,), used | {c}, cum_cost + add_cost))
            
            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]
        
        if not beam:
            return board
        
        best_order = beam[0][0]
        
        # Build new board with this column order
        new_board = {}
        for new_c, orig_c in enumerate(best_order):
            for r in range(gs):
                new_board[(r, new_c)] = columns[orig_c][r]
        
        new_score = self.evaluate(new_board)
        old_score = self.evaluate(board)
        
        if new_score < old_score - 1e-9:
            print(f"        Column reorder improved: {old_score:.4f} -> {new_score:.4f}")
            return new_board
        
        return board

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
        """Compute board score including border penalties (lower = better)."""
        total = 0.0
        count = 0
        gs = self.grid_size
        border_penalty_total = 0.0
        
        for r in range(gs):
            for c in range(gs):
                pid = board.get((r, c))
                if pid is None:
                    continue
                
                # Add border penalty for this piece at this position
                border_penalty_total += self.compat.get_border_penalty(pid, r, c)
                
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
        
        seam_score = total / max(count, 1)
        return seam_score + border_penalty_total
    
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
        Uses precomputed vertical costs for efficiency.
        """
        from itertools import permutations
        
        gs = self.grid_size
        
        # Extract rows
        rows = []
        for r in range(gs):
            row = tuple(board[(r, c)] for c in range(gs))
            rows.append(row)
        
        # Precompute vertical costs between all row pairs
        # vert_cost[i][j] = cost of placing row i above row j
        vert_cost = {}
        for i in range(gs):
            for j in range(gs):
                if i != j:
                    cost = sum(self.compat.get_vertical_score(rows[i][c], rows[j][c]) for c in range(gs))
                    vert_cost[(i, j)] = cost / gs
        
        best_perm = tuple(range(gs))
        best_cost = sum(vert_cost[(best_perm[i], best_perm[i+1])] for i in range(gs-1))
        
        # Try all row permutations (8! = 40320 for 8x8)
        for perm in permutations(range(gs)):
            cost = sum(vert_cost[(perm[i], perm[i+1])] for i in range(gs-1))
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_perm = perm
        
        # Build best board
        best_board = {}
        for new_r, orig_r in enumerate(best_perm):
            for c in range(gs):
                best_board[(new_r, c)] = rows[orig_r][c]
        
        return best_board
    
    def _try_full_column_permutations(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try all permutations of entire columns.
        Uses precomputed horizontal costs for efficiency.
        """
        from itertools import permutations
        
        gs = self.grid_size
        
        # Extract columns
        cols = []
        for c in range(gs):
            col = tuple(board[(r, c)] for r in range(gs))
            cols.append(col)
        
        # Precompute horizontal costs between all column pairs
        # horiz_cost[i][j] = cost of placing column i left of column j
        horiz_cost = {}
        for i in range(gs):
            for j in range(gs):
                if i != j:
                    cost = sum(self.compat.get_horizontal_score(cols[i][r], cols[j][r]) for r in range(gs))
                    horiz_cost[(i, j)] = cost / gs
        
        best_perm = tuple(range(gs))
        best_cost = sum(horiz_cost[(best_perm[i], best_perm[i+1])] for i in range(gs-1))
        
        # Try all column permutations
        for perm in permutations(range(gs)):
            cost = sum(horiz_cost[(perm[i], perm[i+1])] for i in range(gs-1))
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_perm = perm
        
        # Build best board
        best_board = {}
        for new_c, orig_c in enumerate(best_perm):
            for r in range(gs):
                best_board[(r, new_c)] = cols[orig_c][r]
        
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
                if nodes[0] > 300000:
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
                
                for ridx, _ in options[:150]:
                    mask = row_entries[ridx][0]
                    score = row_entries[ridx][2]
                    chosen.append(ridx)
                    dfs(chosen, used_mask | mask, cost + score)
                    chosen.pop()
                    if nodes[0] > 300000:
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
        
        # Focus on corners only (reduced from 4 corners x 64 pieces = 256 to 4 x 16 = 64)
        start_positions = [
            (0, 0), (0, gs-1), (gs-1, 0), (gs-1, gs-1),  # corners
        ]
        
        trials = 0
        # For corners, try top 16 pieces (by border likelihood) as seeds
        for start_row, start_col in start_positions:
            for seed_piece in all_pieces[:16]:  # Only try first 16 pieces
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
        
        # Step 0: Try all cyclic column and row shifts FIRST (fixes global misalignment)
        print("    Trying global cyclic shifts...")
        board = self._try_all_cyclic_shifts(board)
        
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
        
        # Step 5b: Try row cyclic shifts (fixes row misalignment)
        print("    Trying row cyclic shifts...")
        board = self._try_row_cyclic_shifts(board)
        
        # Step 5c: Try column block shifts
        print("    Trying column block shifts...")
        board = self._try_column_block_shifts(board)
        
        # Step 5d: Try all row shift combinations
        print("    Trying global row alignment...")
        board = self._try_all_row_shift_combinations(board)
        
        # Step 5e: Try column reordering by vertical fit
        print("    Trying column reordering...")
        board = self._try_column_reordering_by_vertical_fit(board)
        
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
            # Also try row cyclic shifts
            before = after
            board = self._try_row_cyclic_shifts(board)
            after = self.evaluate(board)
            if abs(after - prev_score) < 1e-9:
                break
            prev_score = after

        final_score = self.evaluate(board)
        print(f"    Final score after cluster refinement: {final_score:.4f}")

        if final_score < initial_score - 1e-6:
            print(f"    IMPROVED: {initial_score:.4f} -> {final_score:.4f}")
        else:
            print(f"    No improvement found")

        # Step 8: Boundary-specific refinement (fixes first/last col, last row issues)
        print("    Boundary-specific refinement...")
        board = self._fix_boundary_pieces(board)
        
        # Step 9: Try rebuilding boundary rows/columns using interior as anchor
        print("    Rebuilding boundary from interior anchor...")
        board = self._rebuild_boundary_from_interior(board)
        
        # Step 10: Try swapping pairs of adjacent pieces (fixes 2-piece group swaps)
        print("    Trying pair swaps...")
        board = self._try_pair_swaps(board)
        
        # Step 11: Iterate boundary fix and rebuild until no improvement
        prev_score = self.evaluate(board)
        for iteration in range(5):
            print(f"    Boundary iteration {iteration + 1}...")
            board = self._fix_boundary_pieces(board)
            board = self._rebuild_boundary_from_interior(board)
            board = self._try_pair_swaps(board)
            board = self._final_global_swap_refinement(board)
            new_score = self.evaluate(board)
            if new_score >= prev_score - 1e-9:
                break
            print(f"      Iteration improved: {prev_score:.4f} -> {new_score:.4f}")
            prev_score = new_score
        
        # Final pass
        board = self._final_global_swap_refinement(board)
        
        return board

    def _try_pair_swaps(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try swapping pairs of adjacent pieces with other pairs.
        This fixes cases where two correctly-grouped pieces need to swap
        positions with another correctly-grouped pair.
        """
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Generate all horizontal pairs (two horizontally adjacent pieces)
        h_pairs = []
        for r in range(gs):
            for c in range(gs - 1):
                h_pairs.append(((r, c), (r, c + 1)))
        
        # Generate all vertical pairs
        v_pairs = []
        for r in range(gs - 1):
            for c in range(gs):
                v_pairs.append(((r, c), (r + 1, c)))
        
        all_pairs = h_pairs + v_pairs
        
        improved = True
        iterations = 0
        while improved and iterations < 3:
            improved = False
            iterations += 1
            
            # Try swapping each pair with every other pair
            for i, pair1 in enumerate(all_pairs):
                for j, pair2 in enumerate(all_pairs):
                    if i >= j:
                        continue
                    
                    # Check if pairs overlap
                    pos1a, pos1b = pair1
                    pos2a, pos2b = pair2
                    if len({pos1a, pos1b, pos2a, pos2b}) < 4:
                        continue  # Pairs overlap, skip
                    
                    # Try swapping the pairs
                    candidate = dict(best_board)
                    # Swap piece at pos1a with piece at pos2a
                    # Swap piece at pos1b with piece at pos2b
                    candidate[pos1a], candidate[pos2a] = candidate[pos2a], candidate[pos1a]
                    candidate[pos1b], candidate[pos2b] = candidate[pos2b], candidate[pos1b]
                    
                    score = self.evaluate(candidate)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_board = candidate
                        improved = True
                        print(f"        Pair swap improved: {self.evaluate(board):.4f} -> {best_score:.4f}")
        
        return best_board

    def _rebuild_boundary_from_interior(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Rebuild each boundary row/column using the adjacent interior as a constraint.
        Uses beam search to find the best piece sequence for each boundary.
        """
        gs = self.grid_size
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Get interior pieces (1 to gs-2 in both dimensions)
        interior_positions = set((r, c) for r in range(1, gs-1) for c in range(1, gs-1))
        interior_pieces = set(board[pos] for pos in interior_positions)
        
        # Boundary pieces are everything else
        boundary_pieces = [pid for pid in range(64) if pid not in interior_pieces]
        
        # Try rebuilding first row using row 1 as anchor
        print("        Rebuilding first row...")
        first_row_board = self._rebuild_row_from_anchor(best_board, 0, 1, boundary_pieces)
        first_row_score = self.evaluate(first_row_board)
        if first_row_score < best_score - 1e-9:
            best_score = first_row_score
            best_board = first_row_board
            print(f"          First row improved: {self.evaluate(board):.4f} -> {best_score:.4f}")
        
        # Try rebuilding last row using row gs-2 as anchor
        print("        Rebuilding last row...")
        last_row_board = self._rebuild_row_from_anchor(best_board, gs-1, gs-2, boundary_pieces)
        last_row_score = self.evaluate(last_row_board)
        if last_row_score < best_score - 1e-9:
            best_score = last_row_score
            best_board = last_row_board
            print(f"          Last row improved: {best_score:.4f}")
        
        # Try rebuilding first column using column 1 as anchor
        print("        Rebuilding first column...")
        first_col_board = self._rebuild_col_from_anchor(best_board, 0, 1, boundary_pieces)
        first_col_score = self.evaluate(first_col_board)
        if first_col_score < best_score - 1e-9:
            best_score = first_col_score
            best_board = first_col_board
            print(f"          First col improved: {best_score:.4f}")
        
        # Try rebuilding last column using column gs-2 as anchor
        print("        Rebuilding last column...")
        last_col_board = self._rebuild_col_from_anchor(best_board, gs-1, gs-2, boundary_pieces)
        last_col_score = self.evaluate(last_col_board)
        if last_col_score < best_score - 1e-9:
            best_score = last_col_score
            best_board = last_col_board
            print(f"          Last col improved: {best_score:.4f}")
        
        return best_board
    
    def _rebuild_row_from_anchor(self, board: Dict[Tuple[int, int], int], 
                                  target_row: int, anchor_row: int,
                                  available_pieces: List[int]) -> Dict[Tuple[int, int], int]:
        """Rebuild a row using the adjacent anchor row as a constraint."""
        gs = self.grid_size
        
        # Get pieces currently in target row
        target_pieces = set(board[(target_row, c)] for c in range(gs))
        
        # Get anchor row pieces
        anchor_pieces = [board[(anchor_row, c)] for c in range(gs)]
        
        # Use beam search to find best assignment
        beam_width = 1000
        
        # State: (row_so_far, used_pieces, score)
        initial = (tuple(), frozenset(), 0.0)
        beam = [initial]
        
        for col in range(gs):
            anchor_pid = anchor_pieces[col]
            new_beam = []
            
            for row_so_far, used, cum_score in beam:
                for pid in target_pieces:
                    if pid in used:
                        continue
                    
                    # Score this placement
                    score = 0.0
                    count = 0
                    
                    # Vertical score with anchor
                    if target_row < anchor_row:
                        score += self.compat.get_vertical_score(pid, anchor_pid)
                    else:
                        score += self.compat.get_vertical_score(anchor_pid, pid)
                    count += 1
                    
                    # Horizontal score with previous piece in row
                    if col > 0:
                        prev_pid = row_so_far[-1]
                        score += self.compat.get_horizontal_score(prev_pid, pid)
                        count += 1
                    
                    avg_score = score / count
                    new_beam.append((row_so_far + (pid,), used | {pid}, cum_score + avg_score))
            
            # Keep top beam_width
            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]
        
        if not beam:
            return board
        
        # Build new board
        best_row = beam[0][0]
        new_board = dict(board)
        for c, pid in enumerate(best_row):
            new_board[(target_row, c)] = pid
        
        return new_board
    
    def _rebuild_col_from_anchor(self, board: Dict[Tuple[int, int], int],
                                  target_col: int, anchor_col: int,
                                  available_pieces: List[int]) -> Dict[Tuple[int, int], int]:
        """Rebuild a column using the adjacent anchor column as a constraint."""
        gs = self.grid_size
        
        # Get pieces currently in target column
        target_pieces = set(board[(r, target_col)] for r in range(gs))
        
        # Get anchor column pieces
        anchor_pieces = [board[(r, anchor_col)] for r in range(gs)]
        
        # Use beam search
        beam_width = 1000
        
        initial = (tuple(), frozenset(), 0.0)
        beam = [initial]
        
        for row in range(gs):
            anchor_pid = anchor_pieces[row]
            new_beam = []
            
            for col_so_far, used, cum_score in beam:
                for pid in target_pieces:
                    if pid in used:
                        continue
                    
                    score = 0.0
                    count = 0
                    
                    # Horizontal score with anchor
                    if target_col < anchor_col:
                        score += self.compat.get_horizontal_score(pid, anchor_pid)
                    else:
                        score += self.compat.get_horizontal_score(anchor_pid, pid)
                    count += 1
                    
                    # Vertical score with previous piece in column
                    if row > 0:
                        prev_pid = col_so_far[-1]
                        score += self.compat.get_vertical_score(prev_pid, pid)
                        count += 1
                    
                    avg_score = score / count
                    new_beam.append((col_so_far + (pid,), used | {pid}, cum_score + avg_score))
            
            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]
        
        if not beam:
            return board
        
        best_col = beam[0][0]
        new_board = dict(board)
        for r, pid in enumerate(best_col):
            new_board[(r, target_col)] = pid
        
        return new_board

    def _fix_boundary_pieces(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Fix boundary pieces by trying all possible assignments for boundary positions.
        If the interior is mostly correct, this can fix edge misplacements.
        """
        from itertools import permutations
        import heapq
        
        gs = self.grid_size
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        # Step 1: Try Hungarian boundary assignment first
        print("      Trying Hungarian boundary assignment...")
        hungarian_board = self._hungarian_boundary_assignment(board)
        hungarian_score = self.evaluate(hungarian_board)
        if hungarian_score < best_score - 1e-9:
            print(f"        Hungarian improved: {best_score:.4f} -> {hungarian_score:.4f}")
            best_board = hungarian_board
            best_score = hungarian_score
        
        board = best_board
        
        # Step 2: Fix individual columns and rows
        # Fix first column
        print("      Fixing first column...")
        board = self._fix_column(board, 0)
        
        # Fix last column
        print("      Fixing last column...")
        board = self._fix_column(board, gs - 1)
        
        # Fix first row
        print("      Fixing first row...")
        board = self._fix_row(board, 0)
        
        # Fix last row
        print("      Fixing last row...")
        board = self._fix_row(board, gs - 1)
        
        # Step 3: Try swapping boundary pieces with any other pieces based on fit
        print("      Trying boundary-to-any swaps...")
        board = self._try_boundary_swaps_with_interior(board)
        
        # Step 4: Try fixing second row/column (often affected by boundary errors)
        print("      Fixing second column...")
        board = self._fix_column(board, 1)
        print("      Fixing second-to-last column...")
        board = self._fix_column(board, gs - 2)
        print("      Fixing second row...")
        board = self._fix_row(board, 1)
        print("      Fixing second-to-last row...")
        board = self._fix_row(board, gs - 2)
        
        # Step 5: Try aggressive reassignment using confident interior as anchors
        print("      Trying aggressive boundary reassignment...")
        board = self._reassign_boundary_from_all(board)
        
        # Step 6: Try permutations of worst-fitting pieces (8 worst)
        print("      Fixing worst-fitting pieces...")
        board = self._fix_worst_pieces(board, num_worst=8)
        
        # Step 7: Targeted swap search on remaining bad pieces
        print("      Targeted swap search...")
        board = self._targeted_swap_search(board)
        
        # One more global pass
        board = self._final_global_swap_refinement(board)
        
        new_score = self.evaluate(board)
        if new_score < best_score - 1e-9:
            print(f"      Boundary fix improved: {best_score:.4f} -> {new_score:.4f}")
            return board
        return best_board
    
    def _fix_column(self, board: Dict[Tuple[int, int], int], col: int) -> Dict[Tuple[int, int], int]:
        """Try all permutations of pieces in a single column using precomputed costs."""
        from itertools import permutations
        
        gs = self.grid_size
        positions = [(r, col) for r in range(gs)]
        pieces = [board[pos] for pos in positions]
        
        # Get fixed neighbors (left and right columns)
        left_neighbors = [board[(r, col-1)] if col > 0 else None for r in range(gs)]
        right_neighbors = [board[(r, col+1)] if col < gs-1 else None for r in range(gs)]
        
        # Precompute costs for each piece at each row position
        # cost[piece_idx][row] = cost of placing pieces[piece_idx] at row
        piece_costs = {}
        for pi, pid in enumerate(pieces):
            for r in range(gs):
                cost = 0.0
                # Vertical neighbors
                if r > 0:
                    cost += self.compat.get_vertical_score(pieces[0], pid)  # placeholder, will use actual
                if r < gs - 1:
                    cost += self.compat.get_vertical_score(pid, pieces[0])  # placeholder
                # Horizontal neighbors (fixed)
                if left_neighbors[r] is not None:
                    cost += self.compat.get_horizontal_score(left_neighbors[r], pid)
                if right_neighbors[r] is not None:
                    cost += self.compat.get_horizontal_score(pid, right_neighbors[r])
                piece_costs[(pi, r)] = cost
        
        best_perm = tuple(range(len(pieces)))
        best_cost = float('inf')
        
        for perm in permutations(range(len(pieces))):
            # Cost = sum of vertical edges + horizontal edges to fixed neighbors
            cost = 0.0
            for r in range(gs):
                pid = pieces[perm[r]]
                # Horizontal to fixed neighbors
                if left_neighbors[r] is not None:
                    cost += self.compat.get_horizontal_score(left_neighbors[r], pid)
                if right_neighbors[r] is not None:
                    cost += self.compat.get_horizontal_score(pid, right_neighbors[r])
                # Vertical to next piece in column
                if r < gs - 1:
                    next_pid = pieces[perm[r+1]]
                    cost += self.compat.get_vertical_score(pid, next_pid)
            
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_perm = perm
        
        # Build best board
        best_board = dict(board)
        for r, pi in enumerate(best_perm):
            best_board[(r, col)] = pieces[pi]
        
        return best_board
    
    def _fix_row(self, board: Dict[Tuple[int, int], int], row: int) -> Dict[Tuple[int, int], int]:
        """Try all permutations of pieces in a single row using precomputed costs."""
        from itertools import permutations
        
        gs = self.grid_size
        positions = [(row, c) for c in range(gs)]
        pieces = [board[pos] for pos in positions]
        
        # Get fixed neighbors (top and bottom rows)
        top_neighbors = [board[(row-1, c)] if row > 0 else None for c in range(gs)]
        bottom_neighbors = [board[(row+1, c)] if row < gs-1 else None for c in range(gs)]
        
        best_perm = tuple(range(len(pieces)))
        best_cost = float('inf')
        
        for perm in permutations(range(len(pieces))):
            # Cost = sum of horizontal edges + vertical edges to fixed neighbors
            cost = 0.0
            for c in range(gs):
                pid = pieces[perm[c]]
                # Vertical to fixed neighbors
                if top_neighbors[c] is not None:
                    cost += self.compat.get_vertical_score(top_neighbors[c], pid)
                if bottom_neighbors[c] is not None:
                    cost += self.compat.get_vertical_score(pid, bottom_neighbors[c])
                # Horizontal to next piece in row
                if c < gs - 1:
                    next_pid = pieces[perm[c+1]]
                    cost += self.compat.get_horizontal_score(pid, next_pid)
            
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_perm = perm
        
        # Build best board
        best_board = dict(board)
        for c, pi in enumerate(best_perm):
            best_board[(row, c)] = pieces[pi]
        
        return best_board

    def _try_boundary_swaps_with_interior(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Try swapping each boundary piece with each interior piece.
        This can fix cases where boundary and interior pieces are swapped.
        """
        gs = self.grid_size
        
        boundary_positions = set()
        for r in range(gs):
            boundary_positions.add((r, 0))
            boundary_positions.add((r, gs - 1))
        for c in range(gs):
            boundary_positions.add((0, c))
            boundary_positions.add((gs - 1, c))
        
        interior_positions = set((r, c) for r in range(gs) for c in range(gs)) - boundary_positions
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        improved = True
        while improved:
            improved = False
            for b_pos in boundary_positions:
                for i_pos in interior_positions:
                    candidate = dict(best_board)
                    candidate[b_pos], candidate[i_pos] = candidate[i_pos], candidate[b_pos]
                    
                    score = self.evaluate(candidate)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_board = candidate
                        improved = True
        
        return best_board

    def _hungarian_boundary_assignment(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Use Hungarian algorithm to optimally assign pieces to boundary positions.
        The interior is kept fixed, and we find the best assignment of remaining
        pieces to boundary positions.
        """
        gs = self.grid_size
        
        # Define boundary positions (edges of the grid)
        boundary_positions = []
        # First row
        for c in range(gs):
            boundary_positions.append((0, c))
        # Last row
        for c in range(gs):
            boundary_positions.append((gs - 1, c))
        # First column (excluding corners already added)
        for r in range(1, gs - 1):
            boundary_positions.append((r, 0))
        # Last column (excluding corners already added)
        for r in range(1, gs - 1):
            boundary_positions.append((r, gs - 1))
        
        boundary_positions = list(set(boundary_positions))  # Remove duplicates
        interior_positions = [(r, c) for r in range(gs) for c in range(gs) 
                              if (r, c) not in boundary_positions]
        
        # Get boundary pieces (pieces currently on boundary)
        boundary_pieces = [board[pos] for pos in boundary_positions]
        interior_pieces_set = set(board[pos] for pos in interior_positions)
        
        n_boundary = len(boundary_positions)
        
        # Build cost matrix: cost[i][j] = cost of placing boundary_pieces[j] at boundary_positions[i]
        cost_matrix = np.zeros((n_boundary, n_boundary))
        
        for i, pos in enumerate(boundary_positions):
            r, c = pos
            for j, pid in enumerate(boundary_pieces):
                cost = 0.0
                count = 0
                
                # Check all 4 neighbors
                neighbors = [
                    (r - 1, c, 'top'),
                    (r + 1, c, 'bottom'),
                    (r, c - 1, 'left'),
                    (r, c + 1, 'right')
                ]
                
                for nr, nc, direction in neighbors:
                    if 0 <= nr < gs and 0 <= nc < gs:
                        neighbor_pos = (nr, nc)
                        if neighbor_pos in interior_positions:
                            # Interior is fixed, use its piece
                            neighbor_pid = board[neighbor_pos]
                            if direction == 'top':
                                cost += self.compat.get_vertical_score(neighbor_pid, pid)
                            elif direction == 'bottom':
                                cost += self.compat.get_vertical_score(pid, neighbor_pid)
                            elif direction == 'left':
                                cost += self.compat.get_horizontal_score(neighbor_pid, pid)
                            elif direction == 'right':
                                cost += self.compat.get_horizontal_score(pid, neighbor_pid)
                            count += 1
                
                cost_matrix[i][j] = cost / max(count, 1)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Build new board
        new_board = dict(board)
        for i, j in zip(row_ind, col_ind):
            new_board[boundary_positions[i]] = boundary_pieces[j]
        
        return new_board

    def _identify_confident_interior(self, board: Dict[Tuple[int, int], int]) -> Set[Tuple[int, int]]:
        """
        Identify interior positions where pieces are confidently placed.
        These are positions where the piece has strong matches with ALL neighbors.
        """
        gs = self.grid_size
        confident = set()
        
        for r in range(1, gs - 1):  # Exclude boundary
            for c in range(1, gs - 1):
                pid = board[(r, c)]
                
                # Check all 4 neighbors exist
                total_score = 0.0
                count = 0
                
                # Left
                left_pid = board[(r, c - 1)]
                total_score += self.compat.get_horizontal_score(left_pid, pid)
                count += 1
                
                # Right
                right_pid = board[(r, c + 1)]
                total_score += self.compat.get_horizontal_score(pid, right_pid)
                count += 1
                
                # Top
                top_pid = board[(r - 1, c)]
                total_score += self.compat.get_vertical_score(top_pid, pid)
                count += 1
                
                # Bottom
                bottom_pid = board[(r + 1, c)]
                total_score += self.compat.get_vertical_score(pid, bottom_pid)
                count += 1
                
                avg_score = total_score / count
                
                # If average score is very good, mark as confident
                if avg_score < 0.10:  # Low score = good match
                    confident.add((r, c))
        
        return confident

    def _reassign_boundary_from_all(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        More aggressive boundary fix: identify the most confident interior pieces,
        keep them fixed, and reassign ALL remaining pieces to fill the rest.
        """
        gs = self.grid_size
        
        # Find confidently placed interior pieces
        confident = self._identify_confident_interior(board)
        print(f"        Found {len(confident)} confidently placed interior pieces")
        
        if len(confident) < 20:  # Not enough confident placements
            return board
        
        # Fixed positions are confident interior
        fixed_positions = confident
        fixed_pieces = {board[pos] for pos in fixed_positions}
        
        # Positions to reassign
        reassign_positions = [(r, c) for r in range(gs) for c in range(gs) 
                              if (r, c) not in fixed_positions]
        reassign_pieces = [pid for pid in range(64) if pid not in fixed_pieces]
        
        n = len(reassign_positions)
        if n != len(reassign_pieces):
            return board  # Sanity check
        
        # Build cost matrix for reassignment
        cost_matrix = np.zeros((n, n))
        
        for i, pos in enumerate(reassign_positions):
            r, c = pos
            for j, pid in enumerate(reassign_pieces):
                cost = 0.0
                count = 0
                
                neighbors = [
                    (r - 1, c, 'top'),
                    (r + 1, c, 'bottom'),
                    (r, c - 1, 'left'),
                    (r, c + 1, 'right')
                ]
                
                for nr, nc, direction in neighbors:
                    if 0 <= nr < gs and 0 <= nc < gs:
                        neighbor_pos = (nr, nc)
                        if neighbor_pos in fixed_positions:
                            neighbor_pid = board[neighbor_pos]
                            if direction == 'top':
                                cost += self.compat.get_vertical_score(neighbor_pid, pid)
                            elif direction == 'bottom':
                                cost += self.compat.get_vertical_score(pid, neighbor_pid)
                            elif direction == 'left':
                                cost += self.compat.get_horizontal_score(neighbor_pid, pid)
                            elif direction == 'right':
                                cost += self.compat.get_horizontal_score(pid, neighbor_pid)
                            count += 1
                
                cost_matrix[i][j] = cost / max(count, 1) if count > 0 else 1.0
        
        # Solve
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Build new board
        new_board = dict(board)
        for i, j in zip(row_ind, col_ind):
            new_board[reassign_positions[i]] = reassign_pieces[j]
        
        new_score = self.evaluate(new_board)
        old_score = self.evaluate(board)
        
        if new_score < old_score - 1e-9:
            print(f"        Aggressive reassign improved: {old_score:.4f} -> {new_score:.4f}")
            return new_board
        return board

    def _fix_worst_pieces(self, board: Dict[Tuple[int, int], int], num_worst: int = 8) -> Dict[Tuple[int, int], int]:
        """
        Find the worst-fitting pieces and try all permutations among them.
        For num_worst=8, this is 8! = 40320 permutations (feasible).
        Uses delta scoring for efficiency.
        """
        from itertools import permutations
        
        gs = self.grid_size
        
        # Calculate local fit score for each piece
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
                
                avg_score = local_score / max(count, 1)
                piece_scores.append(((r, c), avg_score))
        
        # Sort by worst fit (highest score)
        piece_scores.sort(key=lambda x: -x[1])
        
        # Take the worst num_worst pieces (capped at 8 for performance)
        num_worst = min(num_worst, 8)
        worst_positions = [pos for pos, _ in piece_scores[:num_worst]]
        worst_pieces = [board[pos] for pos in worst_positions]
        
        # Precompute neighbor info for affected positions
        fixed_neighbors = {}  # pos -> list of (neighbor_pos, direction)
        for pos in worst_positions:
            r, c = pos
            neighbors = []
            if c > 0 and (r, c-1) not in worst_positions:
                neighbors.append(((r, c-1), 'left'))
            if c < gs-1 and (r, c+1) not in worst_positions:
                neighbors.append(((r, c+1), 'right'))
            if r > 0 and (r-1, c) not in worst_positions:
                neighbors.append(((r-1, c), 'top'))
            if r < gs-1 and (r+1, c) not in worst_positions:
                neighbors.append(((r+1, c), 'bottom'))
            fixed_neighbors[pos] = neighbors
        
        # Internal edges between worst positions
        internal_edges = []
        worst_set = set(worst_positions)
        for i, pos1 in enumerate(worst_positions):
            r1, c1 = pos1
            for j, pos2 in enumerate(worst_positions):
                if j <= i:
                    continue
                r2, c2 = pos2
                if r1 == r2 and abs(c1 - c2) == 1:
                    internal_edges.append((i, j, 'h', min(c1, c2)))
                elif c1 == c2 and abs(r1 - r2) == 1:
                    internal_edges.append((i, j, 'v', min(r1, r2)))
        
        best_perm = tuple(range(len(worst_pieces)))
        best_cost = float('inf')
        
        for perm in permutations(range(len(worst_pieces))):
            cost = 0.0
            # Fixed neighbor costs
            for i, pos in enumerate(worst_positions):
                pid = worst_pieces[perm[i]]
                for npos, direction in fixed_neighbors[pos]:
                    npid = board[npos]
                    if direction == 'left':
                        cost += self.compat.get_horizontal_score(npid, pid)
                    elif direction == 'right':
                        cost += self.compat.get_horizontal_score(pid, npid)
                    elif direction == 'top':
                        cost += self.compat.get_vertical_score(npid, pid)
                    elif direction == 'bottom':
                        cost += self.compat.get_vertical_score(pid, npid)
            
            # Internal edge costs
            for i, j, edge_type, _ in internal_edges:
                pid_i = worst_pieces[perm[i]]
                pid_j = worst_pieces[perm[j]]
                r_i, c_i = worst_positions[i]
                r_j, c_j = worst_positions[j]
                if edge_type == 'h':
                    if c_i < c_j:
                        cost += self.compat.get_horizontal_score(pid_i, pid_j)
                    else:
                        cost += self.compat.get_horizontal_score(pid_j, pid_i)
                else:
                    if r_i < r_j:
                        cost += self.compat.get_vertical_score(pid_i, pid_j)
                    else:
                        cost += self.compat.get_vertical_score(pid_j, pid_i)
            
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_perm = perm
        
        # Build best board
        best_board = dict(board)
        for i, pi in enumerate(best_perm):
            best_board[worst_positions[i]] = worst_pieces[pi]
        
        return best_board

    def _targeted_swap_search(self, board: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        """
        Systematically try swapping each worst piece with every other piece.
        More thorough than random sampling.
        """
        gs = self.grid_size
        
        # Find pieces with poor local fit
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
                
                avg_score = local_score / max(count, 1)
                piece_scores.append(((r, c), avg_score))
        
        # Sort by worst fit
        piece_scores.sort(key=lambda x: -x[1])
        
        # Take worst 20 pieces as candidates for swapping
        worst_positions = [pos for pos, _ in piece_scores[:20]]
        all_positions = [(r, c) for r in range(gs) for c in range(gs)]
        
        best_board = dict(board)
        best_score = self.evaluate(board)
        
        improved = True
        while improved:
            improved = False
            for pos1 in worst_positions:
                for pos2 in all_positions:
                    if pos1 == pos2:
                        continue
                    
                    candidate = dict(best_board)
                    candidate[pos1], candidate[pos2] = candidate[pos2], candidate[pos1]
                    
                    score = self.evaluate(candidate)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_board = candidate
                        improved = True
        
        return best_board


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