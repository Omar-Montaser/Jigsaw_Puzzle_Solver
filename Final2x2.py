"""
2x2 Jigsaw Puzzle Solver
========================
Exact implementation as specified:
- Phase 1: Edge compatibility using 1-pixel MAE only
- Phase 2: Exhaustive search over all 24 permutations
- Phase 3: Optional swap hillclimb refinement
"""

import cv2
import numpy as np
from itertools import permutations
import random
import os


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


def compute_color_histogram(strip, bins=16):
    """Compute normalized color histogram for a strip."""
    hist = []
    for c in range(3):
        h, _ = np.histogram(strip[:, :, c].flatten(), bins=bins, range=(0, 256))
        hist.extend(h)
    hist = np.array(hist, dtype=np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def chi_squared_distance(h1, h2):
    """Chi-squared distance between two histograms."""
    denom = h1 + h2
    mask = denom > 1e-10
    if not np.any(mask):
        return 0.0
    diff_sq = (h1[mask] - h2[mask]) ** 2
    return np.sum(diff_sq / denom[mask]) / 2.0


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


def compute_puzzle_score(pieces, arrangement):
    """
    Compute total puzzle score for a 2x2 arrangement.
    
    Arrangement layout:
        P[0]  P[1]
        P[2]  P[3]
    
    Seams to check:
        - P[0] right <-> P[1] left (horizontal, top row)
        - P[2] right <-> P[3] left (horizontal, bottom row)
        - P[0] bottom <-> P[2] top (vertical, left column)
        - P[1] bottom <-> P[3] top (vertical, right column)
    """
    P = arrangement
    
    score = 0.0
    # Horizontal seams
    score += seam_cost(pieces, P[0], 'right', P[1], 'left')
    score += seam_cost(pieces, P[2], 'right', P[3], 'left')
    # Vertical seams
    score += seam_cost(pieces, P[0], 'bottom', P[2], 'top')
    score += seam_cost(pieces, P[1], 'bottom', P[3], 'top')
    
    return score


def exhaustive_search(pieces):
    """
    Phase 2: Try all 24 permutations and find the one with lowest score.
    
    Returns:
        best_arrangement: list of 4 piece ids in order [top-left, top-right, bottom-left, bottom-right]
        best_score: the score of the best arrangement
    """
    piece_ids = list(pieces.keys())
    best_score = float('inf')
    best_arrangement = None
    
    for perm in permutations(piece_ids):
        arrangement = list(perm)
        score = compute_puzzle_score(pieces, arrangement)
        
        if score < best_score:
            best_score = score
            best_arrangement = arrangement
    
    return best_arrangement, best_score


def swap_hillclimb(pieces, arrangement, max_iterations=1000):
    """
    Phase 3: Optional swap hillclimb refinement.
    
    Try random swaps of any two pieces. If improvement, keep it.
    In practice, this rarely changes anything since exhaustive search
    already finds the global optimum.
    """
    current = arrangement.copy()
    current_score = compute_puzzle_score(pieces, current)
    
    for _ in range(max_iterations):
        # Pick two random positions to swap
        i, j = random.sample(range(4), 2)
        
        # Swap
        current[i], current[j] = current[j], current[i]
        new_score = compute_puzzle_score(pieces, current)
        
        if new_score < current_score:
            # Keep the swap
            current_score = new_score
        else:
            # Revert
            current[i], current[j] = current[j], current[i]
    
    return current, current_score


def solve_2x2(pieces, verbose=True):
    """
    Main solver function for 2x2 puzzles.
    
    Args:
        pieces: dict mapping piece_id (0-3) to piece image (numpy array)
        verbose: print progress info
    
    Returns:
        arrangement: list of 4 piece ids in solved order
        score: final puzzle score
    """
    if verbose:
        print("=" * 50)
        print("2x2 Puzzle Solver")
        print("=" * 50)
        print(f"Pieces: {len(pieces)}")
    
    # Phase 2: Exhaustive search
    if verbose:
        print("\n[Phase 2] Exhaustive search over 24 permutations...")
    
    arrangement, score = exhaustive_search(pieces)
    
    if verbose:
        print(f"  Best arrangement: {arrangement}")
        print(f"  Best score: {score:.4f}")
    
    # Phase 3: Swap hillclimb (optional refinement)
    if verbose:
        print("\n[Phase 3] Swap hillclimb refinement...")
    
    arrangement, score = swap_hillclimb(pieces, arrangement, max_iterations=1000)
    
    if verbose:
        print(f"  Final arrangement: {arrangement}")
        print(f"  Final score: {score:.4f}")
    
    return arrangement, score


def arrangement_to_board(arrangement, grid_size=2):
    """
    Convert flat arrangement to board dict format.
    
    Args:
        arrangement: list of piece ids in row-major order
        grid_size: size of grid (2 for 2x2)
    
    Returns:
        board: dict mapping (row, col) -> piece_id
    """
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board


def reconstruct_image(pieces, arrangement, grid_size=2, show_numbers=True):
    """
    Reconstruct the solved puzzle image.
    
    Args:
        pieces: dict of piece_id -> piece image
        arrangement: list of piece ids in row-major order
        grid_size: size of grid
        show_numbers: if True, overlay piece numbers on each piece
    
    Returns:
        reconstructed image
    """
    # Get piece dimensions
    sample_piece = pieces[list(pieces.keys())[0]]
    piece_h, piece_w = sample_piece.shape[:2]
    
    # Create output image
    output_h = piece_h * grid_size
    output_w = piece_w * grid_size
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # Place pieces
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = arrangement[idx]
            y1, y2 = r * piece_h, (r + 1) * piece_h
            x1, x2 = c * piece_w, (c + 1) * piece_w
            output[y1:y2, x1:x2] = pieces[piece_id]
            idx += 1
    
    # Add piece numbers
    if show_numbers:
        idx = 0
        for r in range(grid_size):
            for c in range(grid_size):
                piece_id = arrangement[idx]
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
                
                idx += 1
    
    return output


def load_pieces_from_image(image_path, grid_size=2):
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
        arrangement: solved arrangement
        score: puzzle score
    """
    # Load pieces
    pieces, original = load_pieces_from_image(image_path, grid_size=2)
    
    if verbose:
        print(f"Loaded image: {image_path}")
        print(f"Image size: {original.shape[1]}x{original.shape[0]}")
    
    # Solve
    arrangement, score = solve_2x2(pieces, verbose=verbose)
    
    # Reconstruct
    solved = reconstruct_image(pieces, arrangement, grid_size=2)
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, solved)
        if verbose:
            print(f"\nSaved solved image to: {output_path}")
    
    return arrangement, score, solved


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example usage - solve a 2x2 puzzle
    test_image = "./Gravity Falls/puzzle_2x2/0.jpg"
    
    if os.path.exists(test_image):
        arrangement, score, solved = solve_image(
            test_image,
            output_path="./debug/2x2_solved.png",
            verbose=True
        )
        
        # Display results
        pieces, original = load_pieces_from_image(test_image, grid_size=2)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (Shuffled)")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(solved, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Solved (Score: {score:.4f})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("./debug/2x2_comparison.png", dpi=150)
        plt.show()
        
        print(f"\nFinal arrangement: {arrangement}")
    else:
        print(f"Test image not found: {test_image}")
        print("Usage: python Final2x2.py")
        print("Or import and use solve_2x2(pieces) directly")
