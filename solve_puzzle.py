#!/usr/bin/env python
"""
Artifact-First Puzzle Solver

Usage:
    python solve_puzzle.py <image_path> [--output <output_path>] [--grid <size>]
    
Examples:
    python solve_puzzle.py "./Gravity Falls/puzzle_2x2/0.jpg"
    python solve_puzzle.py "./Gravity Falls/puzzle_4x4/0.jpg" --output "./debug/solved.png"

Architecture:
    Phase 1: Produce artifacts (edges, enhanced grayscale) - MANDATORY
    Phase 2: Solve using multi-feature matching (edges + texture + RGB)
"""

import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt

from pipeline import solve_and_reconstruct


def main():
    parser = argparse.ArgumentParser(
        description="Artifact-first jigsaw puzzle solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features used for matching:
  - Edges (35%): Canny edge continuity (PRIMARY)
  - Texture (35%): Enhanced grayscale texture (PRIMARY)  
  - RGB (20%): Color matching (SECONDARY)
  - Gradient (10%): Intensity gradient continuity (SECONDARY)
        """
    )
    parser.add_argument("image_path", help="Path to the puzzle image")
    parser.add_argument("--output", "-o", help="Output path for solved image")
    parser.add_argument("--grid", "-g", type=int, choices=[2, 4, 8],
                        help="Grid size (auto-detected if not specified)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--no-display", action="store_true", help="Don't display result")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # Solve (artifact-first pipeline)
    arrangement, score, solved = solve_and_reconstruct(
        args.image_path,
        output_path=args.output,
        grid_size=args.grid,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nArrangement: {arrangement}")
        print(f"Score: {score:.4f}")
    
    # Display
    if not args.no_display:
        original = cv2.imread(args.image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original (Shuffled)")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(solved, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Solved (Score: {score:.4f})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
