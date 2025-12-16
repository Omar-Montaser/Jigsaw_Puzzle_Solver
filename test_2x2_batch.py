"""
Batch test script for 2x2 puzzle solver.
Runs the solver on 20 images and reports results.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from solvers.solver_2x2 import solve_2x2, arrangement_to_board


def load_pieces_from_image(image_path: str, grid_size: int = 2) -> dict:
    """Load an image and split it into puzzle pieces as artifacts."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    piece_h, piece_w = h // grid_size, w // grid_size
    
    artifacts = {}
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * piece_h, (row + 1) * piece_h
            x1, x2 = col * piece_w, (col + 1) * piece_w
            artifacts[idx] = {'rgb': img[y1:y2, x1:x2].copy()}
            idx += 1
    
    return artifacts


def reconstruct_image(artifacts: dict, arrangement: list, grid_size: int = 2) -> np.ndarray:
    """Reconstruct the solved puzzle image."""
    sample = artifacts[0]['rgb']
    piece_h, piece_w = sample.shape[:2]
    
    output = np.zeros((piece_h * grid_size, piece_w * grid_size, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            piece_id = arrangement[idx]
            y1, y2 = r * piece_h, (r + 1) * piece_h
            x1, x2 = c * piece_w, (c + 1) * piece_w
            output[y1:y2, x1:x2] = artifacts[piece_id]['rgb']
            idx += 1
    
    return output


def is_correct(arrangement: list) -> bool:
    """Check if arrangement is correct (pieces in original order)."""
    return arrangement == [0, 1, 2, 3]


def run_batch_test(puzzle_dir: str, num_images: int = 20, output_dir: str = "./debug/2x2_batch"):
    """Run solver on multiple images and report results."""
    
    # Find available images
    puzzle_path = Path(puzzle_dir)
    image_files = sorted(puzzle_path.glob("*.jpg"))[:num_images]
    
    if not image_files:
        print(f"No images found in {puzzle_dir}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"2x2 SOLVER BATCH TEST - {len(image_files)} images")
    print("=" * 70)
    print()
    
    results = []
    correct_count = 0
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        
        try:
            # Load pieces
            artifacts = load_pieces_from_image(str(img_path), grid_size=2)
            
            # Solve (quiet mode)
            arrangement, score = solve_2x2(artifacts, verbose=False)
            
            # Check correctness
            correct = is_correct(arrangement)
            if correct:
                correct_count += 1
            
            status = "✓ CORRECT" if correct else f"✗ WRONG ({arrangement})"
            print(f"   Score: {score:.4f} | {status}")
            
            # Save result image
            solved_img = reconstruct_image(artifacts, arrangement)
            output_path = Path(output_dir) / f"solved_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), solved_img)
            
            results.append({
                'image': img_path.name,
                'arrangement': arrangement,
                'score': score,
                'correct': correct
            })
            
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append({
                'image': img_path.name,
                'arrangement': None,
                'score': None,
                'correct': False,
                'error': str(e)
            })
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images: {len(image_files)}")
    print(f"Correct: {correct_count}/{len(image_files)} ({100*correct_count/len(image_files):.1f}%)")
    
    # Show incorrect ones
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\nIncorrect solutions ({len(incorrect)}):")
        for r in incorrect:
            if 'error' in r:
                print(f"  {r['image']}: ERROR - {r['error']}")
            else:
                print(f"  {r['image']}: {r['arrangement']} (score: {r['score']:.4f})")
    
    print(f"\nOutput images saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import sys
    
    puzzle_dir = sys.argv[1] if len(sys.argv) > 1 else "./Gravity Falls/puzzle_2x2"
    num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    run_batch_test(puzzle_dir, num_images)
