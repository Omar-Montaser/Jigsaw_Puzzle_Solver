"""
4x4 Jigsaw Puzzle Solver
========================
Exact implementation as specified:
- Phase 1: Edge compatibility using 1-pixel MAE only
- Phase 2: Row construction via beam search
- Phase 3: Select 4 disjoint rows via DFS backtracking
- Phase 4: Vertical row stacking (try all 24 permutations)
- Phase 5: Local refinement via swap hillclimb
- Phase 6: Border polishing
"""

import cv2
import numpy as np
from itertools import permutations
import random
import os
from collections import defaultdict


# ============================================================
# PHASE 1: EDGE COMPATIBILITY (1-pixel MAE only)
# ============================================================

def extract_edge(piece, edge):
    """
    Extract 1-pixel-wide edge strip from a piece.
    
    Args:
        piece: numpy array (H, W, 3)
        edge: 'top', 'bottom', 'left', 'right'
    
    Returns:
        1D array of RGB values along the edge
    """
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
    """
    Compute seam cost between two piece edges using Mean Absolute Error (MAE).
    
    Args:
        pieces: dict of piece_id -> piece image
        A: piece id for first piece
        edgeA: edge of piece A ('top', 'bottom', 'left', 'right')
        B: piece id for second piece
        edgeB: edge of piece B
    
    Returns:
        MAE score (lower is better match)
    """
    edge_pixels_A = extract_edge(pieces[A], edgeA)
    edge_pixels_B = extract_edge(pieces[B], edgeB)
    
    return np.mean(np.abs(edge_pixels_A - edge_pixels_B))


class CompatibilityCache:
    """
    Cache for precomputed edge compatibility scores.
    """
    def __init__(self, pieces):
        self.pieces = pieces
        self.cache = {}
        self._precompute()
    
    def _precompute(self):
        """Precompute all edge compatibility scores."""
        piece_ids = list(self.pieces.keys())
        edges = ['top', 'bottom', 'left', 'right']
        
        for a in piece_ids:
            for b in piece_ids:
                if a == b:
                    continue
                for ea in edges:
                    for eb in edges:
                        key = (a, ea, b, eb)
                        self.cache[key] = seam_cost(self.pieces, a, ea, b, eb)
    
    def get(self, A, edgeA, B, edgeB):
        """Get cached seam cost."""
        key = (A, edgeA, B, edgeB)
        if key in self.cache:
            return self.cache[key]
        return seam_cost(self.pieces, A, edgeA, B, edgeB)


# ============================================================
# PHASE 2: ROW CONSTRUCTION (Beam Search)
# ============================================================

def build_candidate_rows(pieces, compat, beam_width=300, max_rows=50):
    """
    Build candidate horizontal rows of length 4 using beam search.
    
    Args:
        pieces: dict of piece_id -> piece image
        compat: CompatibilityCache
        beam_width: number of partial rows to keep at each step
        max_rows: maximum number of unique rows to return
    
    Returns:
        list of (row, score) tuples, sorted by score ascending
    """
    piece_ids = list(pieces.keys())
    
    # Start with each piece as a potential row start
    # Each beam entry: (partial_row, used_pieces_set, cumulative_score)
    beam = []
    for pid in piece_ids:
        beam.append(([pid], {pid}, 0.0))
    
    # Extend rows until length 4
    for step in range(3):  # Need 3 more pieces to complete row
        next_beam = []
        
        for row, used, score in beam:
            last_piece = row[-1]
            
            # Try adding each unused piece
            for candidate in piece_ids:
                if candidate in used:
                    continue
                
                # Cost of connecting last_piece.right -> candidate.left
                edge_cost = compat.get(last_piece, 'right', candidate, 'left')
                new_score = score + edge_cost
                
                new_row = row + [candidate]
                new_used = used | {candidate}
                
                next_beam.append((new_row, new_used, new_score))
        
        # Keep top beam_width entries by score
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
    
    # Extract unique rows with their scores
    seen = set()
    unique_rows = []
    
    for row, _, score in beam:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append((row, score))
    
    # Sort by score and return top max_rows
    unique_rows.sort(key=lambda x: x[1])
    return unique_rows[:max_rows]


def compute_row_score(compat, row):
    """
    Compute the horizontal seam score for a row.
    
    Row score = sum of seam(row[i].right, row[i+1].left) for i=0..2
    """
    score = 0.0
    for i in range(len(row) - 1):
        score += compat.get(row[i], 'right', row[i + 1], 'left')
    return score


# ============================================================
# PHASE 3: SELECT 4 DISJOINT ROWS (DFS Backtracking)
# ============================================================

def select_disjoint_rows(candidate_rows, num_rows=4):
    """
    Select 4 rows that do not share any pieces using DFS backtracking.
    
    Args:
        candidate_rows: list of (row, score) tuples
        num_rows: number of rows to select (4 for 4x4)
    
    Returns:
        best_selection: list of 4 rows
        best_score: sum of row scores
    """
    best_selection = None
    best_score = float('inf')
    
    def dfs(idx, selected, used_pieces, cumulative_score):
        nonlocal best_selection, best_score
        
        # Pruning: if current score already exceeds best, stop
        if cumulative_score >= best_score:
            return
        
        # Found a valid selection
        if len(selected) == num_rows:
            if cumulative_score < best_score:
                best_score = cumulative_score
                best_selection = selected.copy()
            return
        
        # Try adding more rows
        for i in range(idx, len(candidate_rows)):
            row, score = candidate_rows[i]
            row_set = set(row)
            
            # Check if this row shares pieces with already selected rows
            if row_set & used_pieces:
                continue
            
            # Add this row and recurse
            selected.append(row)
            dfs(i + 1, selected, used_pieces | row_set, cumulative_score + score)
            selected.pop()
    
    dfs(0, [], set(), 0.0)
    
    return best_selection, best_score


# ============================================================
# PHASE 4: VERTICAL ROW STACKING
# ============================================================

def compute_vertical_seam_score(compat, rows):
    """
    Compute vertical seam score for stacked rows.
    
    Vertical seams: sum of seam(rows[i][col].bottom, rows[i+1][col].top)
    for i=0..2, col=0..3
    """
    score = 0.0
    for i in range(len(rows) - 1):
        for col in range(4):
            score += compat.get(rows[i][col], 'bottom', rows[i + 1][col], 'top')
    return score


def find_best_row_ordering(compat, rows, row_scores_sum):
    """
    Try all 24 permutations of the 4 rows and find the best vertical stacking.
    
    Args:
        compat: CompatibilityCache
        rows: list of 4 rows
        row_scores_sum: sum of horizontal row scores
    
    Returns:
        best_ordering: list of 4 rows in best order
        best_total_score: horizontal + vertical score
    """
    best_ordering = None
    best_total = float('inf')
    best_orientation = None
    # Try all row permutations and both normal/reversed column orderings
    for perm in permutations(range(4)):
        ordered_rows = [rows[i] for i in perm]
        # Try normal and reversed columns
        for col_flip in [False, True]:
            if col_flip:
                flipped_rows = [list(reversed(row)) for row in ordered_rows]
            else:
                flipped_rows = [list(row) for row in ordered_rows]
            vertical_score = compute_vertical_seam_score(compat, flipped_rows)
            total = row_scores_sum + vertical_score
            if total < best_total:
                best_total = total
                best_ordering = flipped_rows
                best_orientation = (perm, col_flip)
    return best_ordering, best_total


# ============================================================
# PHASE 5: LOCAL REFINEMENT (Swap Hillclimb)
# ============================================================

def rows_to_board(rows):
    """Convert list of rows to board dict."""
    board = {}
    for r, row in enumerate(rows):
        for c, piece_id in enumerate(row):
            board[(r, c)] = piece_id
    return board


def board_to_arrangement(board, grid_size=4):
    """Convert board dict to flat arrangement."""
    arrangement = []
    for r in range(grid_size):
        for c in range(grid_size):
            arrangement.append(board[(r, c)])
    return arrangement


def compute_board_score(compat, board, grid_size=4):
    """
    Compute total puzzle score from board.
    
    Includes all horizontal and vertical seams.
    """
    score = 0.0
    
    # Horizontal seams
    for r in range(grid_size):
        for c in range(grid_size - 1):
            score += compat.get(board[(r, c)], 'right', board[(r, c + 1)], 'left')
    
    # Vertical seams
    for r in range(grid_size - 1):
        for c in range(grid_size):
            score += compat.get(board[(r, c)], 'bottom', board[(r + 1, c)], 'top')
    
    return score


def swap_hillclimb(compat, board, max_iterations=40000, grid_size=4):
    """
    Phase 5: Local refinement via random swap hillclimb.
    
    Args:
        compat: CompatibilityCache
        board: dict mapping (row, col) -> piece_id
        max_iterations: number of swap attempts
        grid_size: puzzle grid size
    
    Returns:
        refined board
        final score
    """
    current_board = board.copy()
    current_score = compute_board_score(compat, current_board, grid_size)
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    improvements = 0
    for iteration in range(max_iterations):
        # Pick two random positions
        pos1, pos2 = random.sample(positions, 2)
        
        # Swap pieces
        current_board[pos1], current_board[pos2] = current_board[pos2], current_board[pos1]
        
        new_score = compute_board_score(compat, current_board, grid_size)
        
        if new_score < current_score:
            current_score = new_score
            improvements += 1
        else:
            # Revert swap
            current_board[pos1], current_board[pos2] = current_board[pos2], current_board[pos1]
    
    return current_board, current_score


# ============================================================
# PHASE 6: BORDER POLISHING
# ============================================================

def get_outer_edge_variance(piece, edge):
    """
    Compute variance of the outer edge (texture smoothness indicator).
    Border pieces typically have smoother outer edges.
    """
    edge_pixels = extract_edge(piece, edge)
    return np.var(edge_pixels)


def get_piece_seam_contribution(compat, board, pos, grid_size=4):
    """
    Compute total seam cost contribution of a piece at given position.
    """
    r, c = pos
    piece_id = board[pos]
    score = 0.0
    
    # Left neighbor
    if c > 0:
        score += compat.get(board[(r, c - 1)], 'right', piece_id, 'left')
    
    # Right neighbor
    if c < grid_size - 1:
        score += compat.get(piece_id, 'right', board[(r, c + 1)], 'left')
    
    # Top neighbor
    if r > 0:
        score += compat.get(board[(r - 1, c)], 'bottom', piece_id, 'top')
    
    # Bottom neighbor
    if r < grid_size - 1:
        score += compat.get(piece_id, 'bottom', board[(r + 1, c)], 'top')
    
    return score


def border_polish(pieces, compat, board, grid_size=4):
    """
    Phase 6: Border polishing heuristics.
    
    1) Border preference: boost border positions for pieces with low texture variance
    2) Worst-piece correction: fix the worst scoring pieces
    """
    current_board = board.copy()
    
    # Heuristic 2: Worst-piece correction
    # Find the 4-6 worst scoring pieces
    piece_scores = []
    for pos in current_board:
        score = get_piece_seam_contribution(compat, current_board, pos, grid_size)
        piece_scores.append((pos, score))
    
    # Sort by score (worst = highest)
    piece_scores.sort(key=lambda x: x[1], reverse=True)
    worst_positions = [ps[0] for ps in piece_scores[:6]]
    
    # Interior positions (not on border)
    interior_positions = [
        (r, c) for r in range(1, grid_size - 1) 
        for c in range(1, grid_size - 1)
    ]
    
    # Try swapping worst pieces with interior pieces
    current_score = compute_board_score(compat, current_board, grid_size)
    
    for worst_pos in worst_positions:
        for interior_pos in interior_positions:
            if worst_pos == interior_pos:
                continue
            
            # Swap
            current_board[worst_pos], current_board[interior_pos] = \
                current_board[interior_pos], current_board[worst_pos]
            
            new_score = compute_board_score(compat, current_board, grid_size)
            
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert
                current_board[worst_pos], current_board[interior_pos] = \
                    current_board[interior_pos], current_board[worst_pos]
    
    return current_board, current_score


# ============================================================
# MAIN SOLVER
# ============================================================

def solve_4x4(pieces, verbose=True):
    """
    Main solver function for 4x4 puzzles.
    
    Args:
        pieces: dict mapping piece_id (0-15) to piece image (numpy array)
        verbose: print progress info
    
    Returns:
        board: dict mapping (row, col) -> piece_id
        arrangement: flat list of piece ids in row-major order
        score: final puzzle score
    """
    if verbose:
        print("=" * 60)
        print("4x4 Puzzle Solver")
        print("=" * 60)
        print(f"Pieces: {len(pieces)}")
    
    # Phase 1: Build compatibility cache
    if verbose:
        print("\n[Phase 1] Computing edge compatibility (MAE)...")
    
    compat = CompatibilityCache(pieces)
    
    if verbose:
        print("  Compatibility matrix computed.")
    
    # Phase 2: Build candidate rows via beam search
    if verbose:
        print("\n[Phase 2] Building candidate rows (beam search)...")
    
    # Increase beam_width and max_rows for robustness
    candidate_rows = build_candidate_rows(pieces, compat, beam_width=2000, max_rows=300)
    
    if verbose:
        print(f"  Generated {len(candidate_rows)} unique candidate rows")
        if candidate_rows:
            print(f"  Best row score: {candidate_rows[0][1]:.4f}")
            print(f"  Worst row score: {candidate_rows[-1][1]:.4f}")
    
    # Phase 3: Select 4 disjoint rows via DFS
    if verbose:
        print("\n[Phase 3] Selecting 4 disjoint rows (DFS backtracking)...")
    
    selected_rows, row_score_sum = select_disjoint_rows(candidate_rows, num_rows=4)
    
    if selected_rows is None:
        raise ValueError("Could not find 4 disjoint rows!")
    
    if verbose:
        print(f"  Selected rows with combined score: {row_score_sum:.4f}")
        for i, row in enumerate(selected_rows):
            print(f"    Row {i}: {row}")
    
    # Phase 4: Vertical row stacking
    if verbose:
        print("\n[Phase 4] Finding best vertical row stacking...")
    
    ordered_rows, total_score = find_best_row_ordering(compat, selected_rows, row_score_sum)
    # Check if the top row should be flipped with the bottom row (to match visual requirement)
    # If not, try flipping vertically
    def rows_match_top_bottom(rows):
        # Returns True if the current top row matches the desired bottom row (from the image)
        # For now, just allow both orientations and pick the best
        return True  # Always allow, since we select best below

    # Try both normal and vertically flipped
    arrangements = [ordered_rows, list(reversed(ordered_rows))]
    arrangement_scores = []
    for arr in arrangements:
        board_tmp = rows_to_board(arr)
        score_tmp = compute_board_score(compat, board_tmp, grid_size=4)
        arrangement_scores.append((score_tmp, arr, board_tmp))
    # Pick the best scoring arrangement
    arrangement_scores.sort(key=lambda x: x[0])
    best_score, best_rows, best_board = arrangement_scores[0]
    if verbose:
        print(f"  Best stacking total score: {total_score:.4f} (before orientation check)")
        print(f"  Best arrangement score after orientation check: {best_score:.4f}")
    board = best_board
    
    # Phase 5: Swap hillclimb refinement
    if verbose:
        print("\n[Phase 5] Local refinement (swap hillclimb, 40000 iterations)...")
    
    board, score = swap_hillclimb(compat, board, max_iterations=40000, grid_size=4)
    
    if verbose:
        print(f"  Score after hillclimb: {score:.4f}")
    
    # Phase 6: Border polishing
    if verbose:
        print("\n[Phase 6] Border polishing...")
    
    board, score = border_polish(pieces, compat, board, grid_size=4)
    
    if verbose:
        print(f"  Final score after polishing: {score:.4f}")
    
    # Convert to arrangement
    arrangement = board_to_arrangement(board, grid_size=4)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Final arrangement: {arrangement}")
        print(f"Final score: {score:.4f}")
        print("=" * 60)
    
    return board, arrangement, score


def reconstruct_image(pieces, board, grid_size=4, show_numbers=True):
    """
    Reconstruct the solved puzzle image.
    
    Args:
        pieces: dict of piece_id -> piece image
        board: dict mapping (row, col) -> piece_id
        grid_size: size of grid
        show_numbers: if True, overlay piece numbers on each piece
    
    Returns:
        reconstructed image
    """
    sample_piece = pieces[list(pieces.keys())[0]]
    piece_h, piece_w = sample_piece.shape[:2]
    
    output_h = piece_h * grid_size
    output_w = piece_w * grid_size
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = board[(r, c)]
            y1, y2 = r * piece_h, (r + 1) * piece_h
            x1, x2 = c * piece_w, (c + 1) * piece_w
            output[y1:y2, x1:x2] = pieces[piece_id]
    
    # Add piece numbers
    if show_numbers:
        for r in range(grid_size):
            for c in range(grid_size):
                piece_id = board[(r, c)]
                # Calculate center of this piece
                center_x = c * piece_w + piece_w // 2
                center_y = r * piece_h + piece_h // 2
                
                # Draw text with outline for visibility
                text = str(piece_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = min(piece_w, piece_h) / 80.0  # Scale based on piece size
                thickness = max(2, int(font_scale * 2))
                
                # Get text size to center it
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
                
                # Draw black outline
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                # Draw white text
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return output


def load_pieces_from_image(image_path, grid_size=4):
    """
    Load an image and split it into puzzle pieces.
    
    Args:
        image_path: path to input image
        grid_size: number of pieces per row/column
    
    Returns:
        pieces: dict mapping piece_id -> piece image
        original_image: the loaded image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    pieces = {}
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            pieces[idx] = img[y1:y2, x1:x2].copy()
            idx += 1
    
    return pieces, img


def solve_image(image_path, output_path=None, verbose=True):
    """
    Complete pipeline: load image, solve puzzle, save result.
    
    Args:
        image_path: path to shuffled puzzle image
        output_path: path to save solved image (optional)
        verbose: print progress
    
    Returns:
        board: solved board
        arrangement: flat arrangement
        score: puzzle score
        solved_image: reconstructed image
    """
    # Load pieces
    pieces, original = load_pieces_from_image(image_path, grid_size=4)
    
    if verbose:
        print(f"Loaded image: {image_path}")
        print(f"Image size: {original.shape[1]}x{original.shape[0]}")
    
    # Solve
    board, arrangement, score = solve_4x4(pieces, verbose=verbose)
    
    # Reconstruct
    solved = reconstruct_image(pieces, board, grid_size=4)
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, solved)
        if verbose:
            print(f"\nSaved solved image to: {output_path}")
    
    return board, arrangement, score, solved


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    # Allow image path as command-line argument
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "./Gravity Falls/puzzle_4x4/0.jpg"

    if os.path.exists(test_image):
        board, arrangement, score, solved = solve_image(
            test_image,
            output_path="./debug/4x4_solved.png",
            verbose=True
        )

        # Display results
        pieces, original = load_pieces_from_image(test_image, grid_size=4)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (Shuffled)")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(solved, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Solved (Score: {score:.4f})")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig("./debug/4x4_comparison.png", dpi=150)
        plt.show()

        print(f"\nFinal arrangement: {arrangement}")
    else:
        print(f"Test image not found: {test_image}")
        print("Usage: python Final4x4.py [image_path]")
        print("Or import and use solve_4x4(pieces) directly")
