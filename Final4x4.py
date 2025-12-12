"""
4x4 Jigsaw Puzzle Solver
========================
Scaled up from the working 2x2 solver using the same principles:
- Phase 1: Build match table using same seam_cost() as 2x2
- Phase 2: Constraint-guided DFS/beam search placement
- Phase 3: Swap hillclimb refinement (same as 2x2)

Uses EXACT same edge extraction and scoring logic from 2x2.
"""

import cv2
import numpy as np
from itertools import permutations
import random
import os
import heapq


# ============================================================
# EDGE EXTRACTION (EXACT COPY FROM 2x2)
# ============================================================

def extract_edge(piece, edge, strip_width=5):
    """
    Extract edge strip from a piece (multiple pixels for robustness).
    
    Args:
        piece: numpy array (H, W, 3)
        edge: 'top', 'bottom', 'left', 'right'
        strip_width: number of pixels to include in the strip
    
    Returns:
        2D array of RGB values along the edge strip
    """
    if edge == 'top':
        return piece[:strip_width, :, :].astype(np.float32)
    elif edge == 'bottom':
        return piece[-strip_width:, :, :].astype(np.float32)
    elif edge == 'left':
        return piece[:, :strip_width, :].astype(np.float32)
    elif edge == 'right':
        return piece[:, -strip_width:, :].astype(np.float32)
    else:
        raise ValueError(f"Unknown edge: {edge}")


# ============================================================
# SEAM COST (EXACT COPY FROM 2x2)
# ============================================================

def seam_cost(pieces, A, edgeA, B, edgeB):
    """
    Compute seam cost between two piece edges using SSD and cross-correlation.
    
    Uses Sum of Squared Differences (SSD) which penalizes large differences more.
    Also uses Normalized Cross-Correlation for robustness.
    
    Args:
        pieces: dict of piece_id -> piece image
        A: piece id for first piece
        edgeA: edge of piece A ('top', 'bottom', 'left', 'right')
        B: piece id for second piece
        edgeB: edge of piece B
    
    Returns:
        Combined score (lower is better match)
    """
    stripA = extract_edge(pieces[A], edgeA, strip_width=10)
    stripB = extract_edge(pieces[B], edgeB, strip_width=10)
    
    # For right-left matching
    if edgeA == 'right' and edgeB == 'left':
        # Get the boundary pixels
        edgeA_pixels = stripA[:, -1, :].astype(np.float32).flatten()
        edgeB_pixels = stripB[:, 0, :].astype(np.float32).flatten()
        
        # Get the full strips near the boundary for better context
        stripA_near = stripA[:, -3:, :].astype(np.float32)  # Last 3 columns of A
        stripB_near = stripB[:, :3, :].astype(np.float32)   # First 3 columns of B
        
    elif edgeA == 'bottom' and edgeB == 'top':
        # Get the boundary pixels
        edgeA_pixels = stripA[-1, :, :].astype(np.float32).flatten()
        edgeB_pixels = stripB[0, :, :].astype(np.float32).flatten()
        
        # Get the full strips near the boundary for better context
        stripA_near = stripA[-3:, :, :].astype(np.float32)  # Last 3 rows of A
        stripB_near = stripB[:3, :, :].astype(np.float32)   # First 3 rows of B
    else:
        edgeA_pixels = stripA.flatten().astype(np.float32)
        edgeB_pixels = stripB.flatten().astype(np.float32)
        stripA_near = stripA.astype(np.float32)
        stripB_near = stripB.astype(np.float32)
    
    # 1. SSD - Sum of Squared Differences (penalizes large differences more)
    ssd_score = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    
    # 2. Normalized Cross-Correlation
    # NCC = 1 means perfect match, -1 means opposite, 0 means uncorrelated
    # We convert to a cost where lower = better
    mean_a = np.mean(edgeA_pixels)
    mean_b = np.mean(edgeB_pixels)
    std_a = np.std(edgeA_pixels) + 1e-10
    std_b = np.std(edgeB_pixels) + 1e-10
    
    ncc = np.mean((edgeA_pixels - mean_a) * (edgeB_pixels - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100  # Convert to cost (0 = perfect match)
    
    # 3. Gradient continuity - the change ACROSS the seam should be smooth
    # compared to the average change WITHIN each piece
    stripA_flat = stripA_near.flatten()
    stripB_flat = stripB_near.flatten()
    
    # Internal variance in each strip
    var_a = np.var(stripA_flat) + 1e-10
    var_b = np.var(stripB_flat) + 1e-10
    
    # Difference at the seam
    seam_diff = np.mean((edgeA_pixels - edgeB_pixels) ** 2)
    
    # If seam difference is much larger than internal variance, it's a bad match
    continuity_score = seam_diff / ((var_a + var_b) / 2)
    
    # Combined score
    return 0.3 * np.sqrt(ssd_score) + 0.3 * ncc_cost + 0.4 * continuity_score * 10


# ============================================================
# PHASE 1: BUILD MATCH TABLE
# ============================================================

def build_match_table(pieces):
    """
    Precompute all pairwise edge matching costs.
    
    Returns:
        match: dict where match[A][B]['right'] = seam_cost(A.right, B.left)
                           match[A][B]['bottom'] = seam_cost(A.bottom, B.top)
    """
    piece_ids = list(pieces.keys())
    match = {a: {b: {} for b in piece_ids} for a in piece_ids}
    
    for A in piece_ids:
        for B in piece_ids:
            if A == B:
                match[A][B]['right'] = float('inf')
                match[A][B]['bottom'] = float('inf')
            else:
                # Cost of placing B to the right of A
                match[A][B]['right'] = seam_cost(pieces, A, 'right', B, 'left')
                # Cost of placing B below A
                match[A][B]['bottom'] = seam_cost(pieces, A, 'bottom', B, 'top')
    
    return match


def compute_adjacent_score(match, A, B, direction):
    """
    Get precomputed score for placing B adjacent to A.
    
    Args:
        match: precomputed match table
        A: piece id
        B: piece id
        direction: 'right' (B is right of A) or 'bottom' (B is below A)
    """
    return match[A][B][direction]


# ============================================================
# SCORE BOARD (Adapted from 2x2 compute_puzzle_score)
# ============================================================

def score_board(match, board, grid_size=4):
    """
    Compute total puzzle score for a board arrangement.
    
    Board layout for 4x4:
        (0,0) (0,1) (0,2) (0,3)
        (1,0) (1,1) (1,2) (1,3)
        (2,0) (2,1) (2,2) (2,3)
        (3,0) (3,1) (3,2) (3,3)
    
    Seams to check:
        - 12 horizontal seams: (r,c).right <-> (r,c+1).left
        - 12 vertical seams: (r,c).bottom <-> (r+1,c).top
    
    Args:
        match: precomputed match table
        board: dict mapping (r, c) -> piece_id
        grid_size: size of grid (4 for 4x4)
    
    Returns:
        Total seam cost (lower is better)
    """
    score = 0.0
    
    # Horizontal seams (12 total for 4x4)
    for r in range(grid_size):
        for c in range(grid_size - 1):
            A = board[(r, c)]
            B = board[(r, c + 1)]
            score += match[A][B]['right']
    
    # Vertical seams (12 total for 4x4)
    for r in range(grid_size - 1):
        for c in range(grid_size):
            A = board[(r, c)]
            B = board[(r + 1, c)]
            score += match[A][B]['bottom']
    
    return score


# ============================================================
# PHASE 2: CONSTRAINT-GUIDED BEAM SEARCH
# ============================================================

def beam_solve(pieces, match, beam_width=10000, grid_size=4):
    """
    Build the board using beam search with constraint-guided placement.
    
    Fill grid row-major order:
        for r in range(4):
            for c in range(4):
    
    At each cell (r,c):
        - If left exists -> must match left neighbor
        - If top exists -> must match top neighbor
        - Combine scores and keep best beam_width partial solutions
    
    Args:
        pieces: dict of piece_id -> piece image
        match: precomputed match table
        beam_width: number of partial solutions to keep
        grid_size: size of grid
    
    Returns:
        best_board: dict mapping (r, c) -> piece_id
        best_score: total seam cost
    """
    piece_ids = list(pieces.keys())
    n_pieces = len(piece_ids)
    
    # State: (board_dict, used_set, cumulative_score)
    # Start with empty board
    initial_state = ({}, frozenset(), 0.0)
    beam = [initial_state]
    
    # Fill positions in row-major order
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    for pos_idx, (r, c) in enumerate(positions):
        next_beam = []
        
        for board, used, cum_score in beam:
            # Try each unused piece at this position
            for pid in piece_ids:
                if pid in used:
                    continue
                
                # Compute cost of placing pid at (r, c)
                placement_cost = 0.0
                
                # Check left neighbor
                if c > 0:
                    left_pid = board[(r, c - 1)]
                    placement_cost += match[left_pid][pid]['right']
                
                # Check top neighbor
                if r > 0:
                    top_pid = board[(r - 1, c)]
                    placement_cost += match[top_pid][pid]['bottom']
                
                # Create new state
                new_board = board.copy()
                new_board[(r, c)] = pid
                new_used = used | {pid}
                new_score = cum_score + placement_cost
                
                next_beam.append((new_board, new_used, new_score))
        
        # Keep best beam_width states
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        if (pos_idx + 1) % grid_size == 0:
            print(f"    Row {(pos_idx + 1) // grid_size} complete: {len(next_beam)} -> {len(beam)} states")
    
    # Return best solution
    best_board, _, best_score = beam[0]
    return best_board, best_score


# ============================================================
# PHASE 3: SWAP HILLCLIMB (EXACT SAME LOGIC AS 2x2)
# ============================================================

def swap_hillclimb(match, board, max_iterations=5000, grid_size=4):
    """
    Swap hillclimb refinement - same logic as 2x2.
    
    Try random swaps of any two pieces. If improvement, keep it.
    
    Args:
        match: precomputed match table
        board: dict mapping (r, c) -> piece_id
        max_iterations: number of swap attempts
        grid_size: size of grid
    
    Returns:
        refined_board: improved board
        final_score: score after refinement
    """
    current = board.copy()
    current_score = score_board(match, current, grid_size)
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    improvements = 0
    for iteration in range(max_iterations):
        # Pick two random positions to swap
        pos1, pos2 = random.sample(positions, 2)
        
        # Swap
        current[pos1], current[pos2] = current[pos2], current[pos1]
        new_score = score_board(match, current, grid_size)
        
        if new_score < current_score:
            # Keep the swap
            current_score = new_score
            improvements += 1
        else:
            # Revert
            current[pos1], current[pos2] = current[pos2], current[pos1]
    
    print(f"    Hillclimb: {improvements} improvements")
    return current, current_score


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
        print("4x4 Puzzle Solver (Scaled from 2x2)")
        print("=" * 60)
        print(f"Pieces: {len(pieces)}")
    
    # Phase 1: Build match table
    if verbose:
        print("\n[Phase 1] Building match table...")
    
    match = build_match_table(pieces)
    
    if verbose:
        print("  Match table computed (16x16x2 = 512 scores)")
    
    # Phase 2: Beam search
    if verbose:
        print("\n[Phase 2] Beam search placement...")
    
    board, score = beam_solve(pieces, match, beam_width=20000, grid_size=4)
    
    if verbose:
        print(f"  Beam search score: {score:.4f}")
    
    # Phase 3: Swap hillclimb refinement
    if verbose:
        print("\n[Phase 3] Swap hillclimb refinement (5000 iterations)...")
    
    board, score = swap_hillclimb(match, board, max_iterations=5000, grid_size=4)
    
    if verbose:
        print(f"  Final score: {score:.4f}")
    
    # Convert board to arrangement
    arrangement = board_to_arrangement(board, grid_size=4)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Final arrangement: {arrangement}")
        print(f"Final score: {score:.4f}")
        print("=" * 60)
    
    return board, arrangement, score


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def board_to_arrangement(board, grid_size=4):
    """Convert board dict to flat arrangement list."""
    arrangement = []
    for r in range(grid_size):
        for c in range(grid_size):
            arrangement.append(board[(r, c)])
    return arrangement


def arrangement_to_board(arrangement, grid_size=4):
    """Convert flat arrangement to board dict."""
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board


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
    
    if show_numbers:
        for r in range(grid_size):
            for c in range(grid_size):
                piece_id = board[(r, c)]
                center_x = c * piece_w + piece_w // 2
                center_y = r * piece_h + piece_h // 2
                
                text = str(piece_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = min(piece_w, piece_h) / 80.0
                thickness = max(2, int(font_scale * 2))
                
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
                
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return output


# ============================================================
# IMAGE LOADING
# ============================================================

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
    pieces, original = load_pieces_from_image(image_path, grid_size=4)
    
    if verbose:
        print(f"Loaded image: {image_path}")
        print(f"Image size: {original.shape[1]}x{original.shape[0]}")
    
    board, arrangement, score = solve_4x4(pieces, verbose=verbose)
    solved = reconstruct_image(pieces, board, grid_size=4)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, solved)
        if verbose:
            print(f"\nSaved solved image to: {output_path}")
    
    return board, arrangement, score, solved


# ============================================================
# VERIFICATION: Compare with correct solution
# ============================================================

def find_correct_arrangement(shuffled_pieces, correct_image_path, grid_size=4):
    """
    Find the correct arrangement by matching shuffled pieces to the correct image.
    """
    correct_img = cv2.imread(correct_image_path)
    if correct_img is None:
        return None
    
    h, w = correct_img.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    arrangement = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            correct_piece = correct_img[y1:y2, x1:x2]
            
            best_match = -1
            best_diff = float('inf')
            
            for pid, piece in shuffled_pieces.items():
                if piece.shape != correct_piece.shape:
                    piece = cv2.resize(piece, (correct_piece.shape[1], correct_piece.shape[0]))
                
                diff = np.mean(np.abs(piece.astype(float) - correct_piece.astype(float)))
                if diff < best_diff:
                    best_diff = diff
                    best_match = pid
            
            arrangement.append(best_match)
    
    return arrangement


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
        
        # Try to verify against correct solution
        import re
        match_obj = re.search(r'(\d+)\.jpg', test_image)
        if match_obj:
            img_num = match_obj.group(1)
            correct_path = f"./Gravity Falls/correct/{img_num}.png"
            if os.path.exists(correct_path):
                pieces, _ = load_pieces_from_image(test_image, grid_size=4)
                correct_arr = find_correct_arrangement(pieces, correct_path)
                if correct_arr:
                    match_table = build_match_table(pieces)
                    correct_board = arrangement_to_board(correct_arr, grid_size=4)
                    correct_score = score_board(match_table, correct_board, grid_size=4)
                    print(f"\nCorrect arrangement: {correct_arr}")
                    print(f"Correct score: {correct_score:.4f}")
                    print(f"Solver score: {score:.4f}")
                    if arrangement == correct_arr:
                        print("✓ PERFECT MATCH!")
                    else:
                        print(f"Difference: {correct_score - score:.4f}")

        
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
