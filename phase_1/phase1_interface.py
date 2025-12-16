"""
Phase 1 Interface - Legacy Compatibility

This file provides backward compatibility with the original interface.
The actual implementation is now in pipeline/artifact_pipeline.py
"""

# Re-export from new locations
from pipeline.artifact_pipeline import produce_artifacts as phase1_process
from features.artifacts import create_artifacts_from_pieces_dict as phase1_process_pieces
