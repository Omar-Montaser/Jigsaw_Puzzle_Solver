"""
Puzzle solvers - Artifact-First Architecture.

All solvers REQUIRE Phase 1 artifacts (edges, gray, rgb).
Artifacts are the PRIMARY signal; RGB is secondary.

Usage:
    from pipeline import produce_artifacts
    from solvers import solve_2x2
    
    artifacts, grid = produce_artifacts("puzzle.jpg")
    arrangement, score = solve_2x2(artifacts)  # Will fail without artifacts
"""
from .seam_cost import (
    seam_cost,
    build_match_table,
    score_board,
    validate_artifacts,
    SeamCostWeights,
    # Legacy baseline (deprecated)
    seam_cost_rgb_only
)
from .solver_2x2 import solve_2x2
from .solver_4x4 import solve_4x4
from .solver_8x8 import solve_8x8
