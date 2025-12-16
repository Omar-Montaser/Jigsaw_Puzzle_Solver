"""
Improved 8x8 Puzzle Solver - Refactored
Pixel MAE + border penalty. Beam search + Hungarian rows + refinement.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import time
import heapq
import random
from itertools import permutations
from scipy.optimize import linear_sum_assignment


@dataclass
class SolverConfig:
    """All configurable parameters."""
    margin_threshold: float = 0.50
    mutual_best_conf_ratio: float = 0.50
    very_confident_ratio: float = 0.10
    sa_t0: float = 0.15
    sa_alpha: float = 0.9995
    sa_max_iters: int = 5000
    sa_time_limit: float = 30.0
    acceptable_threshold: float = 0.12
    strip_width: int = 1
    refinement_passes: int = 3
    border_penalty_weight: float = 0.005
    enable_cluster_refinement: bool = True
    enable_diagnostics: bool = True
    debug_dir: str = "./debug"
    verbose: bool = True


def evaluate_board(board: Dict[Tuple[int, int], int], compat, grid_size: int) -> float:
    """Compute average edge score + border penalties."""
    total, count, border_penalty = 0.0, 0, 0.0
    for row in range(grid_size):
        for col in range(grid_size):
            pid = board.get((row, col))
            if pid is None:
                continue
            border_penalty += compat.get_border_penalty(pid, row, col)
            if col < grid_size - 1 and (row, col + 1) in board:
                total += compat.get_horizontal_score(pid, board[(row, col + 1)])
                count += 1
            if row < grid_size - 1 and (row + 1, col) in board:
                total += compat.get_vertical_score(pid, board[(row + 1, col)])
                count += 1
    return total / max(count, 1) + border_penalty


def total_edge_sum(board: Dict[Tuple[int, int], int], compat, grid_size: int) -> float:
    """Sum of all edge scores + border penalties."""
    total = 0.0
    for r in range(grid_size):
        for c in range(grid_size):
            pid = board[(r, c)]
            total += compat.get_border_penalty(pid, r, c)
            if c < grid_size - 1:
                total += compat.get_horizontal_score(pid, board[(r, c + 1)])
            if r < grid_size - 1:
                total += compat.get_vertical_score(pid, board[(r + 1, c)])
    return total


def border_likelihood(edge_strip: np.ndarray) -> float:
    if edge_strip is None or edge_strip.size == 0:
        return 1.0
    gray = cv2.cvtColor(edge_strip, cv2.COLOR_BGR2GRAY) if len(edge_strip.shape) == 3 else edge_strip.copy()
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.var(np.abs(gx) + np.abs(gy)))


def _get_strip(piece: np.ndarray, edge: str, width: int) -> np.ndarray:
    is_color = len(piece.shape) == 3
    if edge == 'top':
        return piece[:width, :, :] if is_color else piece[:width, :]
    elif edge == 'bottom':
        strip = piece[-width:, :, :] if is_color else piece[-width:, :]
        return strip[::-1]
    elif edge == 'left':
        return piece[:, :width, :] if is_color else piece[:, :width]
    elif edge == 'right':
        strip = piece[:, -width:, :] if is_color else piece[:, -width:]
        return strip[:, ::-1]
    raise ValueError(f"Unknown edge: {edge}")


class EdgeDescriptors:
    def __init__(self, piece: np.ndarray, edge: str, strip_width: int = 1):
        self.edge = edge
        strip = _get_strip(piece, edge, strip_width)
        self.pixel_strip = strip.flatten().astype(np.float32) / 255.0
        h, w = piece.shape[:2]
        bw = min(5, h, w)
        is_color = len(piece.shape) == 3
        if edge == 'top':
            border_strip = piece[:bw, :, :] if is_color else piece[:bw, :]
        elif edge == 'bottom':
            border_strip = piece[-bw:, :, :] if is_color else piece[-bw:, :]
        elif edge == 'left':
            border_strip = piece[:, :bw, :] if is_color else piece[:, :bw]
        else:
            border_strip = piece[:, -bw:, :] if is_color else piece[:, -bw:]
        self.border_likelihood = border_likelihood(border_strip)
        self.border_likelihood_normalized = 0.0


class CompatibilityMatrix:
    def __init__(self, pieces: Dict[int, np.ndarray], config: SolverConfig):
        self.config = config
        self.piece_ids = sorted(pieces.keys())
        self.n_pieces = len(self.piece_ids)
        self.grid_size = int(np.sqrt(self.n_pieces))
        self.edges = ['top', 'bottom', 'left', 'right']
        self.complementary = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        
        if config.verbose:
            print("  Computing edge descriptors...")
        self.descriptors: Dict[Tuple[int, str], EdgeDescriptors] = {}
        for pid in self.piece_ids:
            for edge in self.edges:
                self.descriptors[(pid, edge)] = EdgeDescriptors(pieces[pid], edge, config.strip_width)
        
        self._normalize_border_likelihoods()
        
        if config.verbose:
            print("  Computing pairwise compatibility...")
        self.compatibility: Dict[Tuple[int, str, int, str], float] = {}
        self.best_match: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        self.second_best: Dict[Tuple[int, str], Tuple[int, str, float]] = {}
        self._compute_all()
    
    def _normalize_border_likelihoods(self):
        all_vals = [self.descriptors[(pid, edge)].border_likelihood 
                    for pid in self.piece_ids for edge in self.edges]
        min_val, max_val = min(all_vals), max(all_vals)
        range_val = max_val - min_val if max_val > min_val else 1.0
        if self.config.verbose:
            print(f"  Border likelihood range: [{min_val:.2f}, {max_val:.2f}]")
        for pid in self.piece_ids:
            for edge in self.edges:
                desc = self.descriptors[(pid, edge)]
                desc.border_likelihood_normalized = (desc.border_likelihood - min_val) / range_val
    
    def get_border_penalty(self, pid: int, row: int, col: int) -> float:
        gs = self.grid_size
        weight = self.config.border_penalty_weight
        penalty = 0.0
        for edge, is_border in [('top', row == 0), ('bottom', row == gs - 1),
                                 ('left', col == 0), ('right', col == gs - 1)]:
            likelihood = self.descriptors[(pid, edge)].border_likelihood_normalized
            penalty += weight * likelihood if is_border else weight * (1.0 - likelihood) * 0.5
        return penalty
    
    def _compute_edge_score(self, desc1: EdgeDescriptors, desc2: EdgeDescriptors) -> float:
        p1, p2 = desc1.pixel_strip, desc2.pixel_strip
        min_len = min(len(p1), len(p2))
        return float(np.mean(np.abs(p1[:min_len] - p2[:min_len]))) if min_len > 0 else 1.0
    
    def _compute_all(self):
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
                candidates.sort()
                if candidates:
                    self.best_match[(pid_a, edge_a)] = (candidates[0][1], candidates[0][2], candidates[0][0])
                if len(candidates) >= 2:
                    self.second_best[(pid_a, edge_a)] = (candidates[1][1], candidates[1][2], candidates[1][0])
                else:
                    self.second_best[(pid_a, edge_a)] = (None, None, float('inf'))
    
    def get_score(self, pid_a: int, edge_a: str, pid_b: int, edge_b: str) -> float:
        return self.compatibility.get((pid_a, edge_a, pid_b, edge_b), 1.0)
    
    def get_horizontal_score(self, left_pid: int, right_pid: int) -> float:
        return self.get_score(left_pid, 'right', right_pid, 'left')
    
    def get_vertical_score(self, top_pid: int, bottom_pid: int) -> float:
        return self.get_score(top_pid, 'bottom', bottom_pid, 'top')



@dataclass
class LockedPair:
    piece_a: int
    edge_a: str
    piece_b: int
    edge_b: str
    score: float
    confidence_ratio: float
    margin: float


class ConfidentPairDetector:
    def __init__(self, compat: CompatibilityMatrix, config: SolverConfig):
        self.compat = compat
        self.config = config
        self.locked_pairs: List[LockedPair] = []
        self.superpieces: Dict[int, Set[int]] = {}
        self.piece_to_super: Dict[int, int] = {}
    
    def detect_pairs(self) -> List[LockedPair]:
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
                conf_ratio = best_score / (second_score + 1e-9)
                margin = second_score - best_score
                b_best = self.compat.best_match.get((pid_b, edge_b))
                is_mutual = b_best is not None and b_best[0] == pid_a and b_best[1] == edge_a
                should_lock = ((is_mutual and conf_ratio < cfg.mutual_best_conf_ratio) or
                               margin > cfg.margin_threshold or conf_ratio < cfg.very_confident_ratio)
                if should_lock:
                    pair_key = tuple(sorted([(pid_a, edge_a), (pid_b, edge_b)]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        self.locked_pairs.append(LockedPair(
                            piece_a=pid_a, edge_a=edge_a, piece_b=pid_b, edge_b=edge_b,
                            score=best_score, confidence_ratio=conf_ratio, margin=margin))
        if self.config.verbose:
            print(f"  Detected {len(self.locked_pairs)} locked pairs")
        return self.locked_pairs
    
    def build_superpieces(self) -> Dict[int, Set[int]]:
        parent = {pid: pid for pid in self.compat.piece_ids}
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        for pair in self.locked_pairs:
            px, py = find(pair.piece_a), find(pair.piece_b)
            if px != py:
                parent[px] = py
        groups = defaultdict(set)
        for pid in self.compat.piece_ids:
            groups[find(pid)].add(pid)
        self.superpieces = {idx: members for idx, (_, members) in enumerate(groups.items())}
        self.piece_to_super = {pid: idx for idx, members in self.superpieces.items() for pid in members}
        if self.config.verbose:
            sizes = sorted([len(s) for s in self.superpieces.values()], reverse=True)
            print(f"  Created {len(self.superpieces)} superpieces")
            print(f"  Sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
        return self.superpieces


class PuzzleAssembler:
    def __init__(self, compat: CompatibilityMatrix, detector: ConfidentPairDetector,
                 grid_size: int, config: SolverConfig):
        self.compat = compat
        self.detector = detector
        self.grid_size = grid_size
        self.config = config
        self.board: Dict[Tuple[int, int], int] = {}
        self._hungarian_fell_back_to_beam = False
    
    def evaluate_board(self, board: Optional[Dict] = None) -> float:
        return evaluate_board(board or self.board, self.compat, self.grid_size)
    
    def assemble_beam_search(self, beam_width: int = 10000) -> Dict[Tuple[int, int], int]:
        piece_ids = self.compat.piece_ids
        n, gs = len(piece_ids), self.grid_size
        if self.config.verbose:
            print(f"  Beam search (width={beam_width})...")
        beam = [(tuple([None] * n), frozenset(), 0.0, 0)]
        for pos in range(n):
            row, col = pos // gs, pos % gs
            new_beam = []
            for arr, used, cum_score, num_edges in beam:
                for pid in piece_ids:
                    if pid in used:
                        continue
                    edge_score, edges = 0.0, 0
                    if col > 0 and arr[pos - 1] is not None:
                        edge_score += self.compat.get_horizontal_score(arr[pos - 1], pid)
                        edges += 1
                    if row > 0 and arr[pos - gs] is not None:
                        edge_score += self.compat.get_vertical_score(arr[pos - gs], pid)
                        edges += 1
                    new_arr = list(arr)
                    new_arr[pos] = pid
                    new_beam.append((tuple(new_arr), used | {pid}, cum_score + edge_score, num_edges + edges))
            beam = heapq.nsmallest(beam_width, new_beam, key=lambda s: s[2] / max(s[3], 1))
            if self.config.verbose and (pos + 1) % gs == 0:
                print(f"    Row {(pos + 1) // gs}: {len(new_beam)} -> {len(beam)}")
        best_arr = beam[0][0]
        return {(pos // gs, pos % gs): pid for pos, pid in enumerate(best_arr)}
    
    def assemble_hungarian_rows(self) -> Dict[Tuple[int, int], int]:
        piece_ids = list(self.compat.piece_ids)
        gs = self.grid_size
        if self.config.verbose:
            print("  Hungarian row assembly...")
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
        if self.config.verbose:
            print(f"    Got {len(complete_rows)} candidate rows")
            print("    Phase 2: Selecting best rows...")
        max_pool = min(len(complete_rows), 800)
        pool = complete_rows[:max_pool]
        row_entries = [(sum(1 << int(pid) for pid in row), row, score) for row, score in pool]
        row_entries.sort(key=lambda t: t[2])
        piece_to_rows = {int(pid): [] for pid in piece_ids}
        for idx, (mask, row, score) in enumerate(row_entries):
            for pid in row:
                piece_to_rows[int(pid)].append(idx)
        best_rows = None
        def dfs(chosen, used_mask):
            nonlocal best_rows
            if len(chosen) == gs:
                best_rows = [row_entries[i][1] for i in chosen]
                return True
            remaining = [int(pid) for pid in piece_ids if not (used_mask & (1 << int(pid)))]
            if not remaining:
                return False
            pivot = min(remaining, key=lambda pid: sum(1 for ridx in piece_to_rows[pid] 
                                                        if (row_entries[ridx][0] & used_mask) == 0))
            options = [ridx for ridx in piece_to_rows[pivot] if (row_entries[ridx][0] & used_mask) == 0]
            options.sort(key=lambda ridx: row_entries[ridx][2])
            for ridx in options:
                mask = row_entries[ridx][0]
                if mask & used_mask:
                    continue
                chosen.append(ridx)
                if dfs(chosen, used_mask | mask):
                    return True
                chosen.pop()
            return False
        if not dfs([], 0) or best_rows is None:
            if self.config.verbose:
                print(f"    Warning: Could not select {gs} disjoint rows, falling back to beam search")
            self._hungarian_fell_back_to_beam = True
            return self.assemble_beam_search(10000)
        if self.config.verbose:
            print("    Phase 3: Optimal row ordering...")
        def row_vertical_cost(row_top, row_bottom):
            return sum(self.compat.get_vertical_score(row_top[c], row_bottom[c]) for c in range(gs)) / gs
        best_order, best_cost = None, float('inf')
        for perm in permutations(range(gs)):
            cost = sum(row_vertical_cost(best_rows[perm[i]], best_rows[perm[i+1]]) for i in range(gs - 1))
            if cost < best_cost:
                best_cost, best_order = cost, perm
        if self.config.verbose:
            print(f"    Best stacking cost: {best_cost:.4f}")
        return {(row_idx, col_idx): pid for row_idx, orig_idx in enumerate(best_order) 
                for col_idx, pid in enumerate(best_rows[orig_idx])}
    
    def assemble(self) -> Dict[Tuple[int, int], int]:
        self._hungarian_fell_back_to_beam = False
        board1 = self.assemble_hungarian_rows()
        score1 = self.evaluate_board(board1)
        if self.config.verbose:
            print(f"  Hungarian score: {score1:.4f}")
        if self._hungarian_fell_back_to_beam:
            self.board = board1
            return board1
        board2 = self.assemble_beam_search(3000)
        score2 = self.evaluate_board(board2)
        if self.config.verbose:
            print(f"  Beam search score: {score2:.4f}")
        self.board = board1 if score1 <= score2 else board2
        return self.board



class PuzzleRefiner:
    def __init__(self, compat: CompatibilityMatrix, grid_size: int, config: SolverConfig):
        self.compat = compat
        self.grid_size = grid_size
        self.config = config
        self._total_edges = 2 * grid_size * (grid_size - 1)
    
    def evaluate(self, board: Dict[Tuple[int, int], int]) -> float:
        return evaluate_board(board, self.compat, self.grid_size)
    
    def _total_edge_sum(self, board: Dict[Tuple[int, int], int]) -> float:
        return total_edge_sum(board, self.compat, self.grid_size)
    
    def _swap_delta_sum(self, board: Dict[Tuple[int, int], int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
        gs = self.grid_size
        def piece_after_swap(r, c):
            if (r, c) == a: return board[b]
            if (r, c) == b: return board[a]
            return board[(r, c)]
        affected = set()
        for pos in [a, b]:
            r, c = pos
            if c > 0: affected.add(((r, c - 1), (r, c)))
            if c < gs - 1: affected.add(((r, c), (r, c + 1)))
            if r > 0: affected.add(((r - 1, c), (r, c)))
            if r < gs - 1: affected.add(((r, c), (r + 1, c)))
        old_sum, new_sum = 0.0, 0.0
        ra, ca = a
        rb, cb = b
        pid_a_old, pid_b_old = board[a], board[b]
        old_sum += self.compat.get_border_penalty(pid_a_old, ra, ca)
        old_sum += self.compat.get_border_penalty(pid_b_old, rb, cb)
        new_sum += self.compat.get_border_penalty(pid_b_old, ra, ca)
        new_sum += self.compat.get_border_penalty(pid_a_old, rb, cb)
        for (p1, p2) in affected:
            (r1, c1), (r2, c2) = p1, p2
            pid1_old, pid2_old = board[(r1, c1)], board[(r2, c2)]
            pid1_new, pid2_new = piece_after_swap(r1, c1), piece_after_swap(r2, c2)
            if r1 == r2:
                left, right = (pid1_old, pid2_old) if c1 < c2 else (pid2_old, pid1_old)
                left_new, right_new = (pid1_new, pid2_new) if c1 < c2 else (pid2_new, pid1_new)
                old_sum += self.compat.get_horizontal_score(left, right)
                new_sum += self.compat.get_horizontal_score(left_new, right_new)
            else:
                top, bottom = (pid1_old, pid2_old) if r1 < r2 else (pid2_old, pid1_old)
                top_new, bottom_new = (pid1_new, pid2_new) if r1 < r2 else (pid2_new, pid1_new)
                old_sum += self.compat.get_vertical_score(top, bottom)
                new_sum += self.compat.get_vertical_score(top_new, bottom_new)
        return new_sum - old_sum
    
    def local_swap_refinement(self, board: Dict, passes: int = 3, samples_per_pos: int = 64) -> Dict:
        board = dict(board)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(self._total_edges, 1)
        if self.config.verbose:
            print(f"  Local swap refinement (initial: {current_score:.4f})...")
        positions = list(board.keys())
        for pass_num in range(passes):
            improvements = 0
            for pos1 in positions:
                for _ in range(samples_per_pos):
                    pos2 = random.choice(positions)
                    if pos1 == pos2:
                        continue
                    delta = self._swap_delta_sum(board, pos1, pos2)
                    if delta < -1e-12:
                        board[pos1], board[pos2] = board[pos2], board[pos1]
                        current_sum += delta
                        current_score = current_sum / max(self._total_edges, 1)
                        improvements += 1
            if self.config.verbose:
                print(f"    Pass {pass_num + 1}: score={current_score:.4f}, improvements={improvements}")
            if improvements == 0:
                break
        return board
    
    def _block_swap_2x2(self, board):
        gs = self.grid_size
        if gs < 2:
            return None
        r1, c1 = random.randint(0, gs - 2), random.randint(0, gs - 2)
        r2, c2 = random.randint(0, gs - 2), random.randint(0, gs - 2)
        return None if (r1, c1) == (r2, c2) else ((r1, c1), (r2, c2))
    
    def _apply_block_swap_2x2(self, board, a, b):
        r1, c1 = a
        r2, c2 = b
        coords1 = [(r1, c1), (r1, c1 + 1), (r1 + 1, c1), (r1 + 1, c1 + 1)]
        coords2 = [(r2, c2), (r2, c2 + 1), (r2 + 1, c2), (r2 + 1, c2 + 1)]
        vals1, vals2 = [board[p] for p in coords1], [board[p] for p in coords2]
        for p, v in zip(coords1, vals2): board[p] = v
        for p, v in zip(coords2, vals1): board[p] = v
    
    def random_pair_hillclimb(self, board, max_iters: int = 120000):
        board = dict(board)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(self._total_edges, 1)
        best_score, best_board = current_score, dict(board)
        positions = list(board.keys())
        for it in range(max_iters):
            a, b = random.sample(positions, 2)
            delta_sum = self._swap_delta_sum(board, a, b)
            if delta_sum < -1e-12:
                board[a], board[b] = board[b], board[a]
                current_sum += delta_sum
                current_score = current_sum / max(self._total_edges, 1)
                if current_score < best_score:
                    best_score, best_board = current_score, dict(board)
            if self.config.verbose and it % 20000 == 0:
                print(f"    Hillclimb {it}: current={current_score:.4f}, best={best_score:.4f}")
        return best_board
    
    def simulated_annealing(self, board: Dict) -> Dict:
        cfg = self.config
        board = dict(board)
        current_sum = self._total_edge_sum(board)
        current_score = current_sum / max(self._total_edges, 1)
        best_score, best_board = current_score, dict(board)
        if cfg.verbose:
            print(f"  Simulated annealing (initial: {current_score:.4f})...")
        T = cfg.sa_t0
        positions = list(board.keys())
        start_time = time.time()
        for iteration in range(cfg.sa_max_iters):
            if time.time() - start_time > cfg.sa_time_limit:
                if cfg.verbose:
                    print(f"    Time limit reached at iteration {iteration}")
                break
            if random.random() < 0.85:
                pos1, pos2 = random.sample(positions, 2)
                delta_sum = self._swap_delta_sum(board, pos1, pos2)
                board[pos1], board[pos2] = board[pos2], board[pos1]
                undo = ('swap', pos1, pos2, delta_sum)
            else:
                blocks = self._block_swap_2x2(board)
                if blocks is None:
                    continue
                a, b = blocks
                self._apply_block_swap_2x2(board, a, b)
                undo = ('block', a, b, None)
            if undo[0] == 'swap':
                delta = undo[3] / max(self._total_edges, 1)
                new_score = current_score + delta
            else:
                new_score = self.evaluate(board)
                delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if undo[0] == 'swap':
                    current_sum += undo[3]
                if current_score < best_score:
                    best_score, best_board = current_score, dict(board)
            else:
                if undo[0] == 'swap':
                    board[undo[1]], board[undo[2]] = board[undo[2]], board[undo[1]]
                else:
                    self._apply_block_swap_2x2(board, undo[1], undo[2])
            T *= cfg.sa_alpha
            if cfg.verbose and iteration % 1000 == 0:
                print(f"    Iter {iteration}: T={T:.6f}, current={current_score:.4f}, best={best_score:.4f}")
        if cfg.verbose:
            print(f"  SA best score: {best_score:.4f}")
        return best_board
    
    def refine(self, board: Dict) -> Dict:
        if self.config.verbose:
            print("\n  Starting refinement pipeline...")
        board = self.local_swap_refinement(board, passes=self.config.refinement_passes, samples_per_pos=64)
        board = self.random_pair_hillclimb(board, max_iters=120000)
        score = self.evaluate(board)
        if score > self.config.acceptable_threshold:
            board = self.simulated_annealing(board)
        if self.config.verbose:
            print(f"  Final score after refinement: {self.evaluate(board):.4f}")
        return board



class AmbiguityClusterRefiner:
    def __init__(self, compat: CompatibilityMatrix, grid_size: int, config: SolverConfig):
        self.compat = compat
        self.grid_size = grid_size
        self.config = config
    
    def evaluate(self, board):
        return evaluate_board(board, self.compat, self.grid_size)
    
    def _try_all_cyclic_shifts(self, board):
        gs = self.grid_size
        best_board, best_score = dict(board), self.evaluate(board)
        for row_shift in range(gs):
            for col_shift in range(gs):
                if row_shift == 0 and col_shift == 0:
                    continue
                shifted = {((r + row_shift) % gs, (c + col_shift) % gs): board[(r, c)]
                           for r in range(gs) for c in range(gs)}
                score = self.evaluate(shifted)
                if score < best_score - 1e-9:
                    best_score, best_board = score, shifted
        return best_board
    
    def _try_line_permutations(self, board, is_row):
        gs = self.grid_size
        if is_row:
            lines = [tuple(board[(i, c)] for c in range(gs)) for i in range(gs)]
        else:
            lines = [tuple(board[(r, i)] for r in range(gs)) for i in range(gs)]
        best_board, best_score = dict(board), self.evaluate(board)
        for perm in permutations(range(gs)):
            candidate = {}
            for new_idx, orig_idx in enumerate(perm):
                for i in range(gs):
                    if is_row:
                        candidate[(new_idx, i)] = lines[orig_idx][i]
                    else:
                        candidate[(i, new_idx)] = lines[orig_idx][i]
            score = self.evaluate(candidate)
            if score < best_score - 1e-9:
                best_score, best_board = score, candidate
        return best_board
    
    def _final_global_swap_refinement(self, board):
        gs = self.grid_size
        positions = [(r, c) for r in range(gs) for c in range(gs)]
        best_board, best_score = dict(board), self.evaluate(board)
        improved = True
        while improved:
            improved = False
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    a, b = positions[i], positions[j]
                    swap_board = dict(best_board)
                    swap_board[a], swap_board[b] = swap_board[b], swap_board[a]
                    score = self.evaluate(swap_board)
                    if score < best_score - 1e-9:
                        best_score, best_board = score, swap_board
                        improved = True
        return best_board
    
    def refine(self, board):
        if self.config.verbose:
            print("\n  Ambiguity cluster refinement...")
            print(f"    Initial score: {self.evaluate(board):.4f}")
        board = self._try_all_cyclic_shifts(board)
        if self.config.verbose:
            print("    Trying row permutations...")
        board = self._try_line_permutations(board, is_row=True)
        if self.config.verbose:
            print("    Trying column permutations...")
        board = self._try_line_permutations(board, is_row=False)
        board = self._final_global_swap_refinement(board)
        if self.config.verbose:
            print(f"    Final score: {self.evaluate(board):.4f}")
        return board


def load_puzzle(puzzle_path: str) -> Tuple[Dict[int, np.ndarray], int]:
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
    if config is None:
        config = SolverConfig()
    print("=" * 70)
    print("IMPROVED 8x8 PUZZLE SOLVER")
    print("=" * 70)
    start_time = time.time()
    
    print("\n[Phase 0] Loading and computing descriptors...")
    pieces, grid_size = load_puzzle(puzzle_path)
    print(f"  Loaded {len(pieces)} pieces ({grid_size}x{grid_size})")
    compat = CompatibilityMatrix(pieces, config)
    
    print("\n[Phase 1] Detecting confident pairs...")
    detector = ConfidentPairDetector(compat, config)
    detector.detect_pairs()
    detector.build_superpieces()
    
    print("\n[Phase 2] Assembling puzzle...")
    assembler = PuzzleAssembler(compat, detector, grid_size, config)
    board = assembler.assemble()
    print(f"  Assembly score: {assembler.evaluate_board(board):.4f}")
    
    print("\n[Phase 3] Refining solution...")
    refiner = PuzzleRefiner(compat, grid_size, config)
    board = refiner.refine(board)
    
    if config.enable_cluster_refinement:
        print("\n[Phase 4] Cluster refinement...")
        cluster_refiner = AmbiguityClusterRefiner(compat, grid_size, config)
        board = cluster_refiner.refine(board)
    
    final_score = evaluate_board(board, compat, grid_size)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SOLUTION COMPLETE")
    print("=" * 70)
    print(f"Final Score: {final_score:.4f}")
    print(f"Time: {elapsed:.1f}s")
    arrangement = [board[(r, c)] for r in range(grid_size) for c in range(grid_size)]
    print(f"Arrangement: {arrangement}")
    
    return {'board': board, 'arrangement': tuple(arrangement), 'score': final_score,
            'grid_size': grid_size, 'time': elapsed}


def save_solution_image(pieces: Dict[int, np.ndarray], board: Dict[Tuple[int, int], int], 
                        output_path: str = "./debug/solution.png"):
    """Save assembled puzzle image."""
    gs = int(np.sqrt(len(board)))
    first = pieces[0]
    h, w = first.shape[:2]
    is_color = len(first.shape) == 3
    assembled = np.zeros((h * gs, w * gs, 3) if is_color else (h * gs, w * gs), dtype=np.uint8)
    for (row, col), pid in board.items():
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        assembled[y1:y2, x1:x2] = pieces[pid]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, assembled)
    print(f"  Saved solution to: {output_path}")


if __name__ == "__main__":
    import sys
    puzzle_path = sys.argv[1] if len(sys.argv) > 1 else "./Gravity Falls/puzzle_8x8/0.jpg"
    config = SolverConfig(verbose=True, sa_max_iters=3000, sa_time_limit=20.0,
                          enable_cluster_refinement=True, enable_diagnostics=False)
    pieces, gs = load_puzzle(puzzle_path)
    result = solve_puzzle(puzzle_path, config)
    save_solution_image(pieces, result['board'], "./debug/solution.png")
