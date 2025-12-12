"""
4x4 Jigsaw Puzzle Solver (EXACT SPEC)
=====================================
Implements ONLY the steps described in the provided instructions.
No extra features, no descriptors, no learned methods.
"""

import numpy as np
import cv2
from itertools import permutations
import random

# -----------------------------
# STEP 0 — Load 16 image pieces
# -----------------------------
def load_pieces(image_list):
    assert len(image_list) == 16
    pieces = {i: img for i, img in enumerate(image_list)}
    return pieces

# -----------------------------
# STEP 1 — Precompute ALL seam costs
# -----------------------------
def extract_edge(piece, edge):
    if edge == 'top':
        return piece[0, :, :].flatten().astype(np.float32)
    elif edge == 'bottom':
        return piece[-1, :, :].flatten().astype(np.float32)
    elif edge == 'left':
        return piece[:, 0, :].flatten().astype(np.float32)
    elif edge == 'right':
        return piece[:, -1, :].flatten().astype(np.float32)
    else:
        raise ValueError(f"Unknown edge: {edge}")

def seam_cost(pieces, A, edgeA, B, edgeB):
    stripA = extract_edge(pieces[A], edgeA)
    stripB = extract_edge(pieces[B], edgeB)
    return np.mean(np.abs(stripA - stripB))

def precompute_seam_costs(pieces):
    seam = {}
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            for edgeA, edgeB in [('top','bottom'),('right','left'),('bottom','top'),('left','right')]:
                seam[(i, edgeA, j, edgeB)] = seam_cost(pieces, i, edgeA, j, edgeB)
    return seam

# -----------------------------
# STEP 2 — Build candidate horizontal rows by BEAM SEARCH
# -----------------------------
def build_candidate_rows(pieces, seam, beam_width=3000, per_start=40):
    # Option A: For each starting tile, keep its best K rows (K=per_start)
    all_rows = []
    for start in range(16):
        beam = [([start], {start}, 0.0)]
        for depth in range(3):
            next_beam = []
            for row, used, score in beam:
                last = row[-1]
                for u in range(16):
                    if u in used:
                        continue
                    edge_score = seam[(last, 'right', u, 'left')]
                    new_row = row + [u]
                    new_used = used | {u}
                    next_beam.append((new_row, new_used, score + edge_score))
            next_beam.sort(key=lambda x: x[2])
            beam = next_beam[:beam_width]
        # For this start tile, keep its best per_start rows
        beam.sort(key=lambda x: x[2])
        for row, _, score in beam[:per_start]:
            all_rows.append((tuple(row), score))
    print(f"Generated {len(all_rows)} candidate rows (Option A, {per_start} per start tile).")
    return [(list(r), s) for r, s in all_rows]

# -----------------------------
# STEP 3 — Select 4 disjoint rows by DFS
# -----------------------------
def select_disjoint_rows(candidate_rows):
    best = None
    best_score = float('inf')
    def dfs(idx, selected, used, score):
        nonlocal best, best_score
        if len(selected) == 4:
            if score < best_score:
                best = [row for row, _ in selected]
                best_score = score
            return
        for i in range(idx, len(candidate_rows)):
            row, rscore = candidate_rows[i]
            if set(row) & used:
                continue
            dfs(i+1, selected+[(row, rscore)], used | set(row), score + rscore)
    dfs(0, [], set(), 0.0)
    return best, best_score

# -----------------------------
# STEP 4 — Vertical ordering (try all 24 permutations)
# -----------------------------
def compute_vertical_score(seam, rows):
    score = 0.0
    for i in range(3):
        for c in range(4):
            score += seam[(rows[i][c], 'bottom', rows[i+1][c], 'top')]
    return score

def find_best_vertical_ordering(seam, rows, row_score_sum):
    best = None
    best_total = float('inf')
    for perm in permutations(range(4)):
        ordered = [rows[i] for i in perm]
        vscore = compute_vertical_score(seam, ordered)
        total = row_score_sum + vscore
        if total < best_total:
            best_total = total
            best = ordered
    return best, best_total

# -----------------------------
# STEP 5 — Local refinement (pair swap hillclimb)
# -----------------------------
def board_to_grid(board):
    grid = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            grid[r][c] = board[(r,c)]
    return grid

def grid_to_board(grid):
    board = {}
    for r in range(4):
        for c in range(4):
            board[(r,c)] = grid[r][c]
    return board

def compute_total_score(seam, board):
    score = 0.0
    for r in range(4):
        for c in range(3):
            score += seam[(board[(r,c)], 'right', board[(r,c+1)], 'left')]
    for r in range(3):
        for c in range(4):
            score += seam[(board[(r,c)], 'bottom', board[(r+1,c)], 'top')]
    return score

def swap_hillclimb(seam, board):
    positions = [(r,c) for r in range(4) for c in range(4)]
    best_board = dict(board)
    best_score = compute_total_score(seam, best_board)
    for _ in range(20000):
        a, b = random.sample(positions, 2)
        board[a], board[b] = board[b], board[a]
        score = compute_total_score(seam, board)
        if score < best_score:
            best_score = score
            best_board = dict(board)
        else:
            board[a], board[b] = board[b], board[a]
    return best_board, best_score

# -----------------------------
# STEP 6 — Border polish (simple version)
# -----------------------------
def border_burden(seam, board):
    burdens = {}
    for r in range(4):
        for c in range(4):
            pid = board[(r,c)]
            burden = 0.0
            if r == 0:
                burden += seam.get((pid, 'top', -1, 'none'), 0)
            if r == 3:
                burden += seam.get((pid, 'bottom', -1, 'none'), 0)
            if c == 0:
                burden += seam.get((pid, 'left', -1, 'none'), 0)
            if c == 3:
                burden += seam.get((pid, 'right', -1, 'none'), 0)
            burdens[(r,c)] = burden
    return burdens

def border_polish(seam, board):
    # 1) Identify border positions
    border_pos = [(r,c) for r in [0,3] for c in range(4)] + [(r,c) for r in range(1,3) for c in [0,3]]
    interior_pos = [(r,c) for r in range(1,3) for c in range(1,3)]
    # 2) Compute border burden
    burdens = border_burden(seam, board)
    # 3) Identify 4 worst border pieces
    worst = sorted(border_pos, key=lambda p: burdens[p], reverse=True)[:4]
    # 4) Try swapping with each interior piece
    improved = False
    for w in worst:
        for i in interior_pos:
            board[w], board[i] = board[i], board[w]
            score = compute_total_score(seam, board)
            if score < compute_total_score(seam, board):
                improved = True
            else:
                board[w], board[i] = board[i], board[w]
    return board

# -----------------------------
# MAIN SOLVER
# -----------------------------
def solve_4x4(image_list):
    pieces = load_pieces(image_list)
    seam = precompute_seam_costs(pieces)
    candidate_rows = build_candidate_rows(pieces, seam)
    if len(candidate_rows) < 40:
        print(f"Warning: Only {len(candidate_rows)} candidate rows generated. Trying to proceed anyway.")
    selected_rows, row_score_sum = select_disjoint_rows(candidate_rows)
    if selected_rows is None:
        raise RuntimeError("Could not find 4 disjoint rows from candidate rows. Try increasing beam width or check input tiles.")
    ordered_rows, total_score = find_best_vertical_ordering(seam, selected_rows, row_score_sum)
    # Build board
    board = {}
    for r in range(4):
        for c in range(4):
            board[(r,c)] = ordered_rows[r][c]
    # Local refinement
    board, _ = swap_hillclimb(seam, board)
    # Border polish
    board = border_polish(seam, board)
    return board


if __name__ == "__main__":
    import cv2
    import os
    import numpy as np
    # Path to the 4x4 image 10
    img_path = "./Gravity Falls/puzzle_4x4/10.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    h, w = img.shape[:2]
    tile_h, tile_w = h // 4, w // 4
    tiles = []
    for r in range(4):
        for c in range(4):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tile = img[y1:y2, x1:x2].copy()
            tiles.append(tile)
    board = solve_4x4(tiles)
    # Print the board as a 4x4 grid of piece IDs
    print("Solved board (piece IDs):")
    for r in range(4):
        print([board[(r, c)] for c in range(4)])

    # Reconstruct the solved image
    solved_img = np.zeros_like(img)
    for r in range(4):
        for c in range(4):
            pid = board[(r, c)]
            tile = tiles[pid]
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            solved_img[y1:y2, x1:x2] = tile

    # Show the solved image using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(solved_img, cv2.COLOR_BGR2RGB))
    plt.title("Solved 4x4 Puzzle")
    plt.axis('off')
    plt.show()
