"""
Test script for 8x8 heuristic solver with Pairwise Neighbor Accuracy metric.
"""

import cv2
import numpy as np
from pathlib import Path
from solvers.solver_8x8_heuristic import solve_heuristic_8x8

GRID_SIZE = 8


def load_pieces(image_path):
    """Load image and split into pieces."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    ph, pw = h // GRID_SIZE, w // GRID_SIZE
    
    artifacts = {}
    for idx in range(GRID_SIZE * GRID_SIZE):
        r, c = idx // GRID_SIZE, idx % GRID_SIZE
        artifacts[idx] = {'rgb': img[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()}
    
    return artifacts


def compute_neighbor_accuracy(grid: np.ndarray) -> float:
    """
    Compute Pairwise Neighbor Accuracy for a reconstructed grid.
    
    The metric checks if adjacent pieces in the reconstruction are correct 
    neighbors in the ground truth (regardless of absolute position).
    
    Ground truth: piece_id N belongs at position N (row-major order).
    - Piece 0 at (0,0), Piece 1 at (0,1), ..., Piece cols at (1,0), etc.
    - Perfect 8x8: [[0,1,2,...,7],[8,9,10,...,15],...]
    
    Args:
        grid: 2D numpy array of piece IDs from reconstruction
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    rows, cols = grid.shape
    correct_neighbors = 0
    total_boundaries = 0
    
    for r in range(rows):
        for c in range(cols):
            piece_a = grid[r, c]
            
            # Skip if current cell is empty
            if piece_a == -1:
                continue
            
            # Check RIGHT neighbor
            if c < cols - 1:
                piece_b = grid[r, c + 1]
                total_boundaries += 1
                if piece_b != -1:
                    # Pieces A and B are correct right neighbors if:
                    # - A is not at the right edge of a row (A % cols < cols - 1)
                    # - B is exactly A + 1
                    if piece_a % cols < cols - 1 and piece_b == piece_a + 1:
                        correct_neighbors += 1
            
            # Check BOTTOM neighbor
            if r < rows - 1:
                piece_b = grid[r + 1, c]
                total_boundaries += 1
                if piece_b != -1:
                    # Pieces A and B are correct bottom neighbors if:
                    # - A is not at the bottom row (A // cols < rows - 1)
                    # - B is exactly A + cols
                    if piece_a // cols < rows - 1 and piece_b == piece_a + cols:
                        correct_neighbors += 1
    
    if total_boundaries == 0:
        return 0.0
    
    return correct_neighbors / total_boundaries


def board_to_grid(board: dict) -> np.ndarray:
    """Convert board dict to 2D numpy array."""
    grid = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
    for (r, c), pid in board.items():
        grid[r, c] = pid
    return grid


def reconstruct_image(artifacts: dict, board: dict) -> np.ndarray:
    """Reconstruct solved image from board."""
    sample = artifacts[0]['rgb']
    ph, pw = sample.shape[:2]
    
    out = np.zeros((ph * GRID_SIZE, pw * GRID_SIZE, 3), dtype=np.uint8)
    for (r, c), pid in board.items():
        out[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = artifacts[pid]['rgb']
    
    return out


def main():
    puzzle_dir = Path('./Gravity Falls/puzzle_8x8')
    image_files = sorted(puzzle_dir.glob('*.jpg'), key=lambda x: int(x.stem))[:20]
    
    print(f'Testing {len(image_files)} images with heuristic solver')
    print(f'{"ID":>4} | {"Neighbor Acc":>12} | {"Score":>8} | {"Time":>6}')
    print('-' * 40)
    
    results = []
    
    for img_path in image_files:
        image_id = img_path.stem
        
        # Load and solve
        artifacts = load_pieces(img_path)
        
        import time
        start = time.time()
        board, arrangement, score = solve_heuristic_8x8(artifacts, verbose=False)
        elapsed = time.time() - start
        
        # Convert to grid and compute accuracy
        grid = board_to_grid(board)
        accuracy = compute_neighbor_accuracy(grid)
        
        # Reconstruct and save solved image
        solved_img = reconstruct_image(artifacts, board)
        output_path = Path(f'./debug/heuristic_8x8_{image_id}.png')
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), solved_img)
        
        results.append({
            'id': image_id,
            'accuracy': accuracy,
            'score': score,
            'time': elapsed
        })
        
        print(f'{image_id:>4} | {accuracy*100:>11.1f}% | {score:>8.4f} | {elapsed:>5.1f}s | saved: {output_path.name}')
    
    # Summary
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    
    print('-' * 40)
    print(f'{"AVG":>4} | {avg_accuracy*100:>11.1f}% | {"":>8} | {avg_time:>5.1f}s')
    print(f'\nAverage Neighbor Accuracy: {avg_accuracy*100:.1f}%')


if __name__ == '__main__':
    main()
