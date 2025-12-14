"""Test that all modules can be imported correctly."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_core_imports():
    """Test core module imports."""
    from core import load_image, detect_grid_size, split_image
    from core.image_utils import load_image_bgr, to_grayscale
    from core.grid_detection import compute_spatial_energy
    from core.splitting import split_image_to_dict
    print("✓ core imports OK")


def test_features_imports():
    """Test features module imports."""
    from features import enhance_grayscale, detect_edges, PieceArtifact
    from features.artifacts import create_artifacts, create_artifacts_from_pieces
    from features.enhancement import apply_clahe
    from features.edges import compute_gradient_magnitude
    print("✓ features imports OK")


def test_solvers_imports():
    """Test solvers module imports."""
    from solvers import solve_2x2, solve_4x4, solve_8x8
    from solvers.seam_cost import extract_edge_strip, compute_seam_cost, build_match_table
    from solvers.solver_2x2 import exhaustive_search
    from solvers.solver_4x4 import beam_solve
    print("✓ solvers imports OK")


def test_pipeline_imports():
    """Test pipeline module imports."""
    from pipeline import produce_artifacts, solve_puzzle, solve_and_reconstruct
    from pipeline.artifact_pipeline import load_and_process
    from pipeline.solver_pipeline import reconstruct_image
    print("✓ pipeline imports OK")


def test_visualization_imports():
    """Test visualization module imports."""
    from visualization import display_comparison, display_artifacts
    from visualization.display import display_grid, save_comparison
    print("✓ visualization imports OK")


def test_legacy_imports():
    """Test legacy interface imports."""
    from image_utils import load_image, detect_grid_size, split_image
    from phase1_interface import phase1_process, phase1_process_pieces
    print("✓ legacy imports OK")


if __name__ == "__main__":
    test_core_imports()
    test_features_imports()
    test_solvers_imports()
    test_pipeline_imports()
    test_visualization_imports()
    test_legacy_imports()
    print("\n✓ All imports successful!")
