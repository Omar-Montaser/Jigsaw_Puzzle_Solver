"""
Pipeline orchestration modules.

Artifact-first architecture:
1. produce_artifacts() - MANDATORY Phase 1
2. solve_puzzle() - Phase 2 (requires artifacts)
3. reconstruct_image() - Uses RGB from artifacts
"""
from .artifact_pipeline import (
    produce_artifacts,
    produce_artifacts_from_pieces,
    load_and_produce_artifacts,
    create_artifact
)
from .solver_pipeline import (
    solve_puzzle,
    solve_and_reconstruct,
    solve_image,
    reconstruct_image
)
