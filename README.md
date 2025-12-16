# 🧩 Jigsaw Puzzle Solver

An image processing project that automatically solves square jigsaw puzzles of various sizes (2×2, 4×4, 8×8) using edge matching algorithms and optimization techniques.

## Overview

This project implements a complete pipeline for solving jigsaw puzzles from shuffled images:

1. **Grid Detection** - Automatically detects puzzle grid size using Sobel gradient analysis
2. **Artifact Production** - Extracts features from each piece (RGB, enhanced grayscale, edges, blur)
3. **Puzzle Solving** - Uses specialized solvers for each grid size with beam search and refinement
4. **Reconstruction** - Assembles the final solved image

The solver uses **RGB pixel seam matching** as the primary signal, with additional features for regularization.

## Features

- Automatic grid size detection (2×2, 4×4, 8×8)
- Multiple solver algorithms optimized for each puzzle size
- GUI application for interactive puzzle solving
- Batch testing notebooks for accuracy evaluation
- Visualization tools for debugging and analysis

## Installation

### Requirements

- Python 3.10+
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Pillow (for GUI)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Image-Processing-Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install opencv-python numpy scipy matplotlib pillow
```

## Usage

### GUI Application

Launch the interactive puzzle solver:

```bash
python puzzle_gui.py
```

The GUI allows you to:
- Select a puzzle image file
- Watch the solver progress in real-time
- View the reconstructed result
- See whether the puzzle was successfully solved

### Command Line

Solve a single puzzle:

```bash
python solve_puzzle.py "./Gravity Falls/puzzle_4x4/0.jpg"

# With options
python solve_puzzle.py "./Gravity Falls/puzzle_8x8/0.jpg" --output "./debug/solved.png" --grid 8
```

### Python API

```python
from pipeline import solve_and_reconstruct

# Solve a puzzle (auto-detects grid size)
arrangement, score, solved_image = solve_and_reconstruct(
    "path/to/puzzle.jpg",
    output_path="solved.png",  # optional
    verbose=True
)

print(f"Arrangement: {arrangement}")
print(f"Score: {score:.4f}")  # Lower is better
```

### Using Individual Components

```python
from pipeline import produce_artifacts, solve_puzzle, reconstruct_image

# Step 1: Produce artifacts
artifacts, original, grid_size = load_and_produce_artifacts("puzzle.jpg")

# Step 2: Solve
board, arrangement, score = solve_puzzle(artifacts, grid_size)

# Step 3: Reconstruct
solved = reconstruct_image(artifacts, board, grid_size)
```

## Project Structure

```
Image-Processing-Project/
├── puzzle_gui.py           # GUI application (Tkinter)
├── solve_puzzle.py         # CLI entry point
├── improvements_8x8.py     # Advanced 8x8 solver (standalone)
│
├── core/                   # Core image processing utilities
│   ├── __init__.py
│   ├── grid_detection.py   # Sobel-based grid size detection
│   ├── image_utils.py      # Image loading and manipulation
│   └── splitting.py        # Image splitting into pieces
│
├── features/               # Feature extraction modules
│   ├── __init__.py
│   ├── artifacts.py        # PieceArtifact data model
│   ├── edges.py            # Canny edge detection
│   └── enhancement.py      # CLAHE, bilateral filter, Gaussian blur
│
├── pipeline/               # Pipeline orchestration
│   ├── __init__.py
│   ├── artifact_pipeline.py  # Phase 1: Artifact production
│   └── solver_pipeline.py    # Phase 2: Solving and reconstruction
│
├── solvers/                # Puzzle solving algorithms
│   ├── __init__.py
│   ├── seam_cost.py        # Edge matching cost functions
│   ├── solver_2x2.py       # 2×2 solver (exhaustive search)
│   ├── solver_4x4.py       # 4×4 solver (beam search)
│   ├── solver_8x8.py       # 8×8 solver wrapper
│   ├── solver_8x8_heuristic.py  # Lightweight heuristic solver
│   └── improvements_8x8.py # Full 8×8 solver implementation
│
├── visualization/          # Display and plotting utilities
│   ├── __init__.py
│   └── display.py          # Comparison plots, artifact display
│
├── phase_1/                # Phase 1 development notebooks
│   ├── edge_detection_visualizer.ipynb
│   ├── grid_detection_visualizer.ipynb
│   ├── imageSplit_visualizer.ipynb
│   └── phase1_pipeline.ipynb
│
├── tests/                  # Unit tests
│   └── test_imports.py     # Import verification
│
├── trials/                 # Experimental solver implementations
│   ├── Final2x2.py
│   └── Final4x4.py
│
├── Gravity Falls/          # Test dataset
│   ├── correct/            # Ground truth images (0-109.png)
│   ├── puzzle_2x2/         # 2×2 shuffled puzzles (0-109.jpg)
│   ├── puzzle_4x4/         # 4×4 shuffled puzzles (0-109.jpg)
│   └── puzzle_8x8/         # 8×8 shuffled puzzles (0-109.jpg)
│
├── debug/                  # Debug output images
├── outputs/                # Solver output results
├── processed_artifacts/    # Cached artifact data
│
├── test_2x2_batch.ipynb    # 2×2 batch testing notebook
├── test_4x4_batch.ipynb    # 4×4 batch testing notebook
├── test_8x8_batch.ipynb    # 8×8 batch testing (first 50 images)
├── test_8x8_batch_2.ipynb  # 8×8 batch testing (images 50-109)
├── test_accuracy_debug.py  # Accuracy metric development
└── README.md
```

## Algorithms

### Grid Detection (`core/grid_detection.py`)

Uses Sobel gradient profiles to detect grid lines:
- Computes horizontal and vertical gradient energy
- Evaluates partition scores for 2×2, 4×4, and 8×8 hypotheses
- Achieves ~97% accuracy on the test dataset

### Artifact Production (`pipeline/artifact_pipeline.py`)

Produces four feature channels for each piece:
- **RGB**: Original color image (for reconstruction)
- **Gray**: Enhanced grayscale (CLAHE + bilateral filter + Gaussian blur)
- **Edges**: Canny edge map with morphological closing
- **Blur**: Low-frequency appearance (heavy Gaussian blur)

### Seam Cost (`solvers/seam_cost.py`)

Computes edge matching cost using:
- **SSD**: Sum of Squared Differences on boundary pixels
- **NCC**: Normalized Cross-Correlation for texture alignment
- **Continuity**: Seam difference relative to local variance

Formula: `0.3 * sqrt(SSD) + 0.3 * NCC_cost + 0.4 * continuity * 10`

### 2×2 Solver (`solvers/solver_2x2.py`)

- Exhaustive search over all 24 permutations
- Border constraint to break ties on uniform backgrounds
- Swap hillclimb refinement

### 4×4 Solver (`solvers/solver_4x4.py`)

- Beam search with width 20,000
- RGB seams as primary signal
- Border constraint during search
- Global coherence as tie-breaker
- Swap hillclimb refinement

### 8×8 Solver (`improvements_8x8.py`)

Multi-phase approach:
1. **Phase 0**: Precompute descriptors and pairwise compatibility
2. **Phase 1**: Detect confident pairs via mutual best matching
3. **Phase 2**: Hungarian row assembly + beam search
4. **Phase 3**: Local swap refinement + simulated annealing
5. **Phase 4**: Ambiguity cluster permutation search
6. **Phase 5**: Boundary-specific refinement

Features:
- DFS with node limits and timeout guards
- Diverse row generation
- 2D global construction
- Targeted swap search for worst-fitting pieces

## Test Dataset

The `Gravity Falls/` directory contains 110 test images (0-109):
- `correct/`: Ground truth solved images (.png)
- `puzzle_2x2/`: 2×2 shuffled puzzles (.jpg)
- `puzzle_4x4/`: 4×4 shuffled puzzles (.jpg)
- `puzzle_8x8/`: 8×8 shuffled puzzles (.jpg)

## Batch Testing

Jupyter notebooks for evaluating solver accuracy:

```bash
# Run Jupyter
jupyter notebook

# Open test notebooks:
# - test_2x2_batch.ipynb
# - test_4x4_batch.ipynb
# - test_8x8_batch.ipynb
# - test_8x8_batch_2.ipynb
```

Accuracy is computed by comparing each piece position in the solved image against the ground truth using NCC (threshold 0.95).

## Configuration

The 8×8 solver supports extensive configuration via `SolverConfig`:

```python
from improvements_8x8 import solve_puzzle, SolverConfig

config = SolverConfig(
    verbose=True,           # Print progress
    global_time_limit=150,  # Max solve time (seconds)
    sa_max_iters=5000,      # Simulated annealing iterations
    beam_seed_width=200,    # Beam search width
)

result = solve_puzzle("puzzle.jpg", config)
```

## Output Format

Solvers return:
- **board**: Dict mapping `(row, col)` → `piece_id`
- **arrangement**: Flat list of piece IDs in row-major order
- **score**: Final seam cost (lower is better)

Example for a 2×2 puzzle:
```python
board = {(0,0): 2, (0,1): 0, (1,0): 3, (1,1): 1}
arrangement = [2, 0, 3, 1]
score = 45.67
```

## Debug Output

The solver generates diagnostic images in `debug/`:
- `locked_pairs_matrix.png`: Confident pair detection
- `compatibility_horizontal.png`: Horizontal edge costs
- `compatibility_vertical.png`: Vertical edge costs
- `confidence_histogram.png`: Match confidence distribution
- `solution.png`: Final solved puzzle

## Acknowledgments

Test images are from the animated series "Gravity Falls" and are used for educational purposes only.
