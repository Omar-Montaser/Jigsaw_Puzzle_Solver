# Jigsaw Puzzle Solver

Automated puzzle assembly using classical computer vision techniques.

## Project Structure

```
project_root/
├── core/                       # Low-level image operations
│   ├── image_utils.py          # Image loading, conversion
│   ├── grid_detection.py       # Grid size detection
│   └── splitting.py            # Image → patches
│
├── features/                   # Feature extraction
│   ├── enhancement.py          # CLAHE, bilateral, blur
│   ├── edges.py                # Canny + morphology
│   └── artifacts.py            # Artifact data model
│
├── pipeline/                   # Orchestration
│   ├── artifact_pipeline.py    # Produces artifacts (Phase 1)
│   └── solver_pipeline.py      # Consumes artifacts (Phase 2)
│
├── solvers/                    # Puzzle solving algorithms
│   ├── seam_cost.py            # Edge + texture matching
│   ├── solver_2x2.py           # Exhaustive search
│   ├── solver_4x4.py           # Beam search
│   └── solver_8x8.py           # Advanced solver
│
├── visualization/              # Display utilities
│   └── display.py              # Comparison, grid display
│
├── outputs/                    # Generated outputs
│   ├── artifacts/              # Cached artifacts
│   ├── results/                # Solved puzzles
│   └── debug/                  # Debug images
│
├── experiments/                # Jupyter notebooks
├── tests/                      # Test suite
│
├── solve_puzzle.py             # Main CLI entry point
├── Final2x2.py                 # Legacy 2x2 interface
├── Final4x4.py                 # Legacy 4x4 interface
└── improvements_8x8.py         # Advanced 8x8 solver
```

## Quick Start

### Command Line
```bash
# Solve a puzzle
python solve_puzzle.py "./Gravity Falls/puzzle_2x2/0.jpg"

# With output path
python solve_puzzle.py "./Gravity Falls/puzzle_4x4/0.jpg" -o "./outputs/results/solved.png"

# Specify grid size
python solve_puzzle.py "puzzle.jpg" --grid 4
```

### Python API
```python
from pipeline import solve_and_reconstruct

# Complete pipeline
arrangement, score, solved_image = solve_and_reconstruct(
    "puzzle.jpg",
    output_path="solved.png",
    verbose=True
)
```

### Step-by-Step
```python
from pipeline import produce_artifacts, solve_puzzle
from pipeline.solver_pipeline import reconstruct_image

# Phase 1: Produce artifacts
artifacts, grid_size = produce_artifacts("puzzle.jpg")

# Phase 2: Solve
board, arrangement, score = solve_puzzle(artifacts, grid_size)

# Reconstruct
solved = reconstruct_image(artifacts, board, grid_size)
```

## Architecture

### Phase 1: Artifact Production
Produces structured artifacts for each puzzle piece:
- **RGB**: Original color image (for reconstruction)
- **Gray**: Enhanced grayscale (CLAHE + bilateral filter)
- **Edges**: Canny edge map (for edge continuity matching)

### Phase 2: Puzzle Solving
Consumes artifacts and solves the puzzle:
- **2x2**: Exhaustive search (24 permutations)
- **4x4**: Beam search with hillclimb refinement
- **8x8**: Advanced solver with confident pair detection

### Seam Cost Function
Combines edge continuity and texture continuity:
```
cost = 0.7 * edge_cost + 0.3 * texture_cost
```
- Edge cost: MAE of Canny edge boundaries
- Texture cost: MSE of enhanced grayscale boundaries

## Key Features

- **Automatic grid detection**: Detects 2x2, 4x4, or 8x8 puzzles
- **Artifact-based matching**: Uses enhanced features, not raw pixels
- **Modular design**: Solvers are independent of feature extraction
- **Extensible**: Easy to add new solvers or features

## Legacy Compatibility

The old interface still works:
```python
from Final2x2 import solve_image
arrangement, score, solved = solve_image("puzzle.jpg")
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy

## Course Information

**Course**: CSE483 / CESS5004 – Computer Vision  
**Project**: Jigsaw Puzzle Solver using Classical CV Techniques
