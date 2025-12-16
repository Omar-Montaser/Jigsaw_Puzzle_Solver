"""
Batch test script for 2x2 puzzle solver with proper Pairwise Neighbor Accuracy.

This version does NOT assume piece IDs encode ground-truth positions.
Ground truth is derived by matching shuffled pieces to the correct image.
Accuracy is computed based on relative adjacency, not absolute position.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def compute_pairwise_neighbor_accuracy(
    reconstructed_grid: np.ndarray,
    ground_truth_labels: Dict[int, Tuple[int, int]]
) -> float:
    """
    Compute Pairwise Neighbor Accuracy for a reconstructed puzzle grid.
    
    This metric measures how many adjacent pairs in the reconstruction are
    actually neighbors in the ground truth, regardless of absolute position.
    
    Args:
        reconstructed_grid: 2D numpy array of piece IDs, with -1 for empty cells.
        ground_truth_labels: Dict mapping piece_id -> (row, col) in the solved puzzle.
    
    Returns:
        Score in [0.0, 1.0] representing the fraction of correct neighbor pairs.
        Returns 0.0 if no valid pairs exist.
    
    Notes:
        - Only horizontal (right) and vertical (bottom) adjacencies are checked.
        - Pairs involving empty cells (-1) are ignored.
        - Ground-truth neighbors are derived from the labels, not from piece IDs.
    """
    rows, cols = reconstructed_grid.shape
    
    # Build ground-truth neighbor sets from labels
    # Invert labels to get position -> piece_id mapping
    pos_to_piece = {pos: pid for pid, pos in ground_truth_labels.items()}
    
    gt_right_neighbors = {}  # piece_id -> right_neighbor_id or None
    gt_bottom_neighbors = {}  # piece_id -> bottom_neighbor_id or None
    
    for piece_id, (r, c) in ground_truth_labels.items():
        gt_right_neighbors[piece_id] = pos_to_piece.get((r, c + 1))
        gt_bottom_neighbors[piece_id] = pos_to_piece.get((r + 1, c))
    
    correct_pairs = 0
    total_pairs = 0
    
    # Check horizontal pairs (right adjacency)
    for r in range(rows):
        for c in range(cols - 1):
            left_piece = reconstructed_grid[r, c]
            right_piece = reconstructed_grid[r, c + 1]
            
            if left_piece == -1 or right_piece == -1:
                continue
            
            total_pairs += 1
            if gt_right_neighbors.get(left_piece) == right_piece:
                correct_pairs += 1
    
    # Check vertical pairs (bottom adjacency)
    for r in range(rows - 1):
        for c in range(cols):
            top_piece = reconstructed_grid[r, c]
            bottom_piece = reconstructed_grid[r + 1, c]
            
            if top_piece == -1 or bottom_piece == -1:
                continue
            
            total_pairs += 1
            if gt_bottom_neighbors.get(top_piece) == bottom_piece:
                correct_pairs += 1
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0


def extract_pieces(image: np.ndarray, grid_size: int) -> List[np.ndarray]:
    """Extract pieces from an image in row-major order."""
    h, w = image.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    pieces = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            pieces.append(image[y1:y2, x1:x2].copy())
    return pieces


def match_piece_to_position(piece: np.ndarray, correct_pieces: List[np.ndarray]) -> int:
    """Find which position in correct_pieces best matches the given piece."""
    best_idx = 0
    best_score = float('inf')
    
    for idx, correct_piece in enumerate(correct_pieces):
        # Resize if needed
        if piece.shape != correct_piece.shape:
            correct_piece = cv2.resize(correct_piece, (piece.shape[1], piece.shape[0]))
        
        # L2 distance
        diff = np.sum((piece.astype(float) - correct_piece.astype(float)) ** 2)
        if diff < best_score:
            best_score = diff
            best_idx = idx
    
    return best_idx


def load_puzzle_with_ground_truth(
    puzzle_path: str,
    correct_path: str,
    grid_size: int = 2
) -> Tuple[dict, Dict[int, Tuple[int, int]]]:
    """
    Load shuffled puzzle pieces and derive ground truth labels by matching
    to the correct (unshuffled) image.
    
    Args:
        puzzle_path: Path to the shuffled puzzle image.
        correct_path: Path to the correct (unshuffled) image.
        grid_size: Number of rows/cols in the puzzle.
    
    Returns:
        artifacts: Dict of piece_id -> {'rgb': image_array}
        ground_truth_labels: Dict of piece_id -> (row, col) derived by matching
    """
    puzzle_img = cv2.imread(puzzle_path)
    correct_img = cv2.imread(correct_path)
    
    if puzzle_img is None:
        raise ValueError(f"Could not load puzzle image: {puzzle_path}")
    if correct_img is None:
        raise ValueError(f"Could not load correct image: {correct_path}")
    
    # Extract pieces from both images
    puzzle_pieces = extract_pieces(puzzle_img, grid_size)
    correct_pieces = extract_pieces(correct_img, grid_size)
    
    # Build artifacts dict (piece_id is just extraction order from puzzle)
    artifacts = {}
    for idx, piece in enumerate(puzzle_pieces):
        artifacts[idx] = {'rgb': piece}
    
    # Match each puzzle piece to its correct position
    ground_truth_labels = {}
    used_positions = set()
    
    # Compute all match scores
    scores = []
    for piece_id, piece in enumerate(puzzle_pieces):
        for pos_idx, correct_piece in enumerate(correct_pieces):
            if correct_piece.shape != piece.shape:
                correct_piece = cv2.resize(correct_piece, (piece.shape[1], piece.shape[0]))
            diff = np.sum((piece.astype(float) - correct_piece.astype(float)) ** 2)
            scores.append((diff, piece_id, pos_idx))
    
    # Greedy assignment: best matches first
    scores.sort()
    assigned_pieces = set()
    
    for diff, piece_id, pos_idx in scores:
        if piece_id in assigned_pieces or pos_idx in used_positions:
            continue
        
        row, col = pos_idx // grid_size, pos_idx % grid_size
        ground_truth_labels[piece_id] = (row, col)
        assigned_pieces.add(piece_id)
        used_positions.add(pos_idx)
        
        if len(assigned_pieces) == len(puzzle_pieces):
            break
    
    return artifacts, ground_truth_labels


def arrangement_to_grid(arrangement: List[int], grid_size: int = 2) -> np.ndarray:
    """Convert flat arrangement list to 2D grid array."""
    return np.array(arrangement).reshape(grid_size, grid_size)


def reconstruct_image(artifacts: dict, arrangement: List[int], grid_size: int = 2) -> np.ndarray:
    """Reconstruct the solved puzzle image from arrangement."""
    sample = artifacts[0]['rgb']
    piece_h, piece_w = sample.shape[:2]
    
    output = np.zeros((piece_h * grid_size, piece_w * grid_size, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = arrangement[idx]
            if piece_id != -1:
                y1, y2 = r * piece_h, (r + 1) * piece_h
                x1, x2 = c * piece_w, (c + 1) * piece_w
                output[y1:y2, x1:x2] = artifacts[piece_id]['rgb']
            idx += 1
    
    return output


def run_batch_test(
    puzzle_dir: str,
    correct_dir: str,
    num_images: int = 20,
    output_dir: str = "./debug/2x2_batch_v2",
    grid_size: int = 2
):
    """Run solver on multiple images and report results with pairwise neighbor accuracy."""
    from solvers.solver_2x2 import solve_2x2
    
    puzzle_path = Path(puzzle_dir)
    correct_path = Path(correct_dir)
    
    # Find matching puzzle/correct pairs
    puzzle_files = sorted(puzzle_path.glob("*.jpg"), key=lambda p: int(p.stem))[:num_images]
    
    if not puzzle_files:
        print(f"No images found in {puzzle_dir}")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"2x2 SOLVER BATCH TEST (Pairwise Neighbor Accuracy) - {len(puzzle_files)} images")
    print("=" * 70)
    print()
    
    results = []
    total_accuracy = 0.0
    perfect_count = 0
    
    for i, puzzle_file in enumerate(puzzle_files):
        print(f"[{i+1}/{len(puzzle_files)}] {puzzle_file.name}")
        
        # Find corresponding correct image
        correct_file = correct_path / f"{puzzle_file.stem}.png"
        if not correct_file.exists():
            correct_file = correct_path / f"{puzzle_file.stem}.jpg"
        
        if not correct_file.exists():
            print(f"   SKIP: No matching correct image found")
            continue
        
        try:
            # Load pieces with ground truth derived from correct image
            artifacts, ground_truth_labels = load_puzzle_with_ground_truth(
                str(puzzle_file), str(correct_file), grid_size
            )
            
            # Solve
            arrangement, solver_score = solve_2x2(artifacts, verbose=False)
            
            # Convert to grid and compute pairwise neighbor accuracy
            reconstructed_grid = arrangement_to_grid(arrangement, grid_size)
            accuracy = compute_pairwise_neighbor_accuracy(reconstructed_grid, ground_truth_labels)
            
            total_accuracy += accuracy
            if accuracy == 1.0:
                perfect_count += 1
            
            status = "✓ PERFECT" if accuracy == 1.0 else f"Accuracy: {accuracy:.2%}"
            print(f"   Solver score: {solver_score:.4f} | {status}")
            
            # Save result image
            solved_img = reconstruct_image(artifacts, arrangement)
            output_path = Path(output_dir) / f"solved_{puzzle_file.stem}.jpg"
            cv2.imwrite(str(output_path), solved_img)
            
            results.append({
                'image': puzzle_file.name,
                'arrangement': arrangement,
                'solver_score': solver_score,
                'neighbor_accuracy': accuracy,
                'ground_truth_labels': ground_truth_labels
            })
            
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image': puzzle_file.name,
                'arrangement': None,
                'solver_score': None,
                'neighbor_accuracy': 0.0,
                'error': str(e)
            })
    
    # Summary
    n_tested = len([r for r in results if 'error' not in r])
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images tested: {n_tested}")
    print(f"Perfect reconstructions: {perfect_count}/{n_tested} ({100*perfect_count/n_tested:.1f}%)" if n_tested > 0 else "N/A")
    print(f"Mean pairwise neighbor accuracy: {total_accuracy/n_tested:.2%}" if n_tested > 0 else "N/A")
    
    # Show imperfect ones
    imperfect = [r for r in results if r.get('neighbor_accuracy', 0) < 1.0 and 'error' not in r]
    if imperfect:
        print(f"\nImperfect solutions ({len(imperfect)}):")
        for r in imperfect:
            print(f"  {r['image']}: {r['arrangement']} (accuracy: {r['neighbor_accuracy']:.2%})")
    
    print(f"\nOutput images saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import sys
    
    puzzle_dir = sys.argv[1] if len(sys.argv) > 1 else "./Gravity Falls/puzzle_2x2"
    correct_dir = sys.argv[2] if len(sys.argv) > 2 else "./Gravity Falls/correct"
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    run_batch_test(puzzle_dir, correct_dir, num_images)
