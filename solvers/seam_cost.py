"""
Hierarchical seam cost computation.

ARCHITECTURE:
- PRIMARY: RGB continuity (same as legacy solver) - main ranking signal
- REGULARIZERS: Phase 1 artifacts (edges, gray, blur) - penalties/refinements
- GLOBAL: Consistency penalties applied only on complete boards

Artifacts are MANDATORY but serve to REGULARIZE RGB, not override it.

Scoring hierarchy:
1. Beam search ranking: RGB cost (legacy algorithm)
2. Candidate refinement: RGB + artifact penalties
3. Final evaluation: Full score with global consistency
"""

import numpy as np
from typing import Dict, Union
from dataclasses import dataclass


@dataclass
class SeamCostWeights:
    """
    Hierarchical weights for seam cost.
    
    RGB is PRIMARY (baseline behavior).
    Artifacts REGULARIZE the RGB signal.
    """
    # Primary signal (legacy RGB behavior)
    rgb_weight: float = 0.70       # RGB continuity - PRIMARY ranking signal
    
    # Regularization penalties (artifacts)
    lowfreq_weight: float = 0.15   # Low-frequency blur - smoothness regularizer
    edge_weight: float = 0.10      # Edge continuity - structure regularizer
    texture_weight: float = 0.05   # Texture - fine detail regularizer
    
    def __post_init__(self):
        total = self.rgb_weight + self.lowfreq_weight + self.edge_weight + self.texture_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Default weights - RGB primary with artifact regularization
DEFAULT_WEIGHTS = SeamCostWeights()

# Pure RGB weights (for beam search ranking - matches legacy exactly)
RGB_ONLY_WEIGHTS = SeamCostWeights(rgb_weight=1.0, lowfreq_weight=0.0, edge_weight=0.0, texture_weight=0.0)


def validate_artifact(artifact: dict, piece_id: int) -> None:
    """
    Validate that an artifact contains all required Phase 1 outputs.
    Raises ValueError if artifacts are missing or invalid.
    """
    required_keys = ['rgb', 'gray', 'edges', 'blur']
    
    if not isinstance(artifact, dict):
        raise TypeError(f"Piece {piece_id}: Expected artifact dict, got {type(artifact)}")
    
    for key in required_keys:
        if key not in artifact:
            raise ValueError(
                f"Piece {piece_id}: Missing required artifact '{key}'. "
                f"Phase 1 artifacts are mandatory. Run artifact pipeline first."
            )
        if artifact[key] is None:
            raise ValueError(f"Piece {piece_id}: Artifact '{key}' is None")
    
    # Validate shapes
    rgb_shape = artifact['rgb'].shape
    gray_shape = artifact['gray'].shape
    edges_shape = artifact['edges'].shape
    blur_shape = artifact['blur'].shape
    
    if len(rgb_shape) != 3 or rgb_shape[2] != 3:
        raise ValueError(f"Piece {piece_id}: RGB must be (H, W, 3), got {rgb_shape}")
    
    if gray_shape != rgb_shape[:2]:
        raise ValueError(f"Piece {piece_id}: Gray shape {gray_shape} doesn't match RGB {rgb_shape[:2]}")
    
    if edges_shape != rgb_shape[:2]:
        raise ValueError(f"Piece {piece_id}: Edges shape {edges_shape} doesn't match RGB {rgb_shape[:2]}")
    
    if blur_shape != rgb_shape[:2]:
        raise ValueError(f"Piece {piece_id}: Blur shape {blur_shape} doesn't match RGB {rgb_shape[:2]}")


def validate_artifacts(artifacts: Dict[int, dict]) -> None:
    """Validate all artifacts in a collection."""
    if not artifacts:
        raise ValueError("No artifacts provided. Phase 1 artifacts are mandatory.")
    
    for piece_id, artifact in artifacts.items():
        validate_artifact(artifact, piece_id)


def extract_edge_strip(artifact: dict, edge: str, strip_width: int = 5) -> dict:
    """
    Extract edge strips from all artifact channels.
    
    Args:
        artifact: dict with 'rgb', 'gray', 'edges', 'blur' keys (MANDATORY)
        edge: 'top', 'bottom', 'left', 'right'
        strip_width: number of pixels to include
    
    Returns:
        dict with strips from each channel
    """
    strips = {}
    
    for key in ['rgb', 'gray', 'edges', 'blur']:
        src = artifact[key]
        
        if edge == 'top':
            strips[key] = src[:strip_width, ...].astype(np.float32)
        elif edge == 'bottom':
            strips[key] = src[-strip_width:, ...].astype(np.float32)
        elif edge == 'left':
            strips[key] = src[:, :strip_width, ...].astype(np.float32)
        elif edge == 'right':
            strips[key] = src[:, -strip_width:, ...].astype(np.float32)
        else:
            raise ValueError(f"Unknown edge: {edge}")
    
    return strips


def compute_lowfreq_cost(strips_a: dict, strips_b: dict, edge_a: str, edge_b: str) -> float:
    """
    Compute low-frequency appearance cost (PRIMARY FEATURE).
    
    Measures global/smooth appearance continuity across the seam.
    This captures overall intensity regions and helps match pieces
    based on their position in the original image's color/brightness gradient.
    
    Uses the same algorithm as legacy RGB solver but on blurred artifact.
    """
    blur_a = strips_a['blur']
    blur_b = strips_b['blur']
    
    # Get boundary pixels
    if edge_a == 'right' and edge_b == 'left':
        boundary_a = blur_a[:, -1].flatten()
        boundary_b = blur_b[:, 0].flatten()
        near_a = blur_a[:, -3:].flatten()
        near_b = blur_b[:, :3].flatten()
    elif edge_a == 'bottom' and edge_b == 'top':
        boundary_a = blur_a[-1, :].flatten()
        boundary_b = blur_b[0, :].flatten()
        near_a = blur_a[-3:, :].flatten()
        near_b = blur_b[:3, :].flatten()
    else:
        boundary_a = blur_a.flatten()
        boundary_b = blur_b.flatten()
        near_a = blur_a.flatten()
        near_b = blur_b.flatten()
    
    # SSD on boundary (primary signal for low-freq matching)
    ssd = np.mean((boundary_a - boundary_b) ** 2)
    
    # NCC for correlation (how well do the smooth regions align)
    mean_a, mean_b = np.mean(boundary_a), np.mean(boundary_b)
    std_a, std_b = np.std(boundary_a) + 1e-10, np.std(boundary_b) + 1e-10
    ncc = np.mean((boundary_a - mean_a) * (boundary_b - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100  # Scale similar to legacy
    
    # Continuity: seam difference relative to local variance
    var_a = np.var(near_a) + 1e-10
    var_b = np.var(near_b) + 1e-10
    seam_diff = np.mean((boundary_a - boundary_b) ** 2)
    continuity = seam_diff / ((var_a + var_b) / 2)
    
    # Same formula as legacy RGB solver: 0.3*sqrt(ssd) + 0.3*ncc_cost + 0.4*continuity*10
    return 0.3 * np.sqrt(ssd) + 0.3 * ncc_cost + 0.4 * continuity * 10


def compute_edge_cost(strips_a: dict, strips_b: dict, edge_a: str, edge_b: str) -> float:
    """
    Compute edge continuity cost using Canny edge maps (PRIMARY FEATURE).
    
    Measures how well edge structures align across the seam.
    """
    edges_a = strips_a['edges']
    edges_b = strips_b['edges']
    
    # Get boundary pixels
    if edge_a == 'right' and edge_b == 'left':
        boundary_a = edges_a[:, -1].flatten()
        boundary_b = edges_b[:, 0].flatten()
    elif edge_a == 'bottom' and edge_b == 'top':
        boundary_a = edges_a[-1, :].flatten()
        boundary_b = edges_b[0, :].flatten()
    else:
        boundary_a = edges_a.flatten()
        boundary_b = edges_b.flatten()
    
    # MAE on edge maps
    edge_mae = np.mean(np.abs(boundary_a - boundary_b))
    
    # Edge density similarity (structural consistency)
    density_a = np.mean(edges_a > 0)
    density_b = np.mean(edges_b > 0)
    density_diff = abs(density_a - density_b) * 100
    
    return 0.7 * edge_mae + 0.3 * density_diff


def compute_texture_cost(strips_a: dict, strips_b: dict, edge_a: str, edge_b: str) -> float:
    """
    Compute texture continuity cost using enhanced grayscale (PRIMARY FEATURE).
    
    Measures intensity/texture continuity across the seam.
    """
    gray_a = strips_a['gray']
    gray_b = strips_b['gray']
    
    # Get boundary pixels
    if edge_a == 'right' and edge_b == 'left':
        boundary_a = gray_a[:, -1].flatten()
        boundary_b = gray_b[:, 0].flatten()
        near_a = gray_a[:, -3:].flatten()
        near_b = gray_b[:, :3].flatten()
    elif edge_a == 'bottom' and edge_b == 'top':
        boundary_a = gray_a[-1, :].flatten()
        boundary_b = gray_b[0, :].flatten()
        near_a = gray_a[-3:, :].flatten()
        near_b = gray_b[:3, :].flatten()
    else:
        boundary_a = gray_a.flatten()
        boundary_b = gray_b.flatten()
        near_a = gray_a.flatten()
        near_b = gray_b.flatten()
    
    # SSD on boundary
    ssd = np.mean((boundary_a - boundary_b) ** 2)
    
    # NCC for texture correlation
    mean_a, mean_b = np.mean(boundary_a), np.mean(boundary_b)
    std_a, std_b = np.std(boundary_a) + 1e-10, np.std(boundary_b) + 1e-10
    ncc = np.mean((boundary_a - mean_a) * (boundary_b - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 50
    
    # Gradient continuity
    var_a = np.var(near_a) + 1e-10
    var_b = np.var(near_b) + 1e-10
    seam_diff = np.mean((boundary_a - boundary_b) ** 2)
    continuity = seam_diff / ((var_a + var_b) / 2)
    
    return 0.3 * np.sqrt(ssd) + 0.4 * ncc_cost + 0.3 * continuity * 5


def compute_rgb_cost_legacy(strips_a: dict, strips_b: dict, edge_a: str, edge_b: str) -> float:
    """
    Compute RGB cost using EXACT legacy algorithm from Final4x4.py.
    
    This is the PRIMARY ranking signal - matches baseline behavior exactly.
    Formula: 0.3 * sqrt(ssd) + 0.3 * ncc_cost + 0.4 * continuity * 10
    """
    rgb_a = strips_a['rgb']
    rgb_b = strips_b['rgb']
    
    # Get boundary and near-boundary pixels (same as legacy)
    if edge_a == 'right' and edge_b == 'left':
        boundary_a = rgb_a[:, -1, :].flatten()
        boundary_b = rgb_b[:, 0, :].flatten()
        near_a = rgb_a[:, -3:, :].flatten()
        near_b = rgb_b[:, :3, :].flatten()
    elif edge_a == 'bottom' and edge_b == 'top':
        boundary_a = rgb_a[-1, :, :].flatten()
        boundary_b = rgb_b[0, :, :].flatten()
        near_a = rgb_a[-3:, :, :].flatten()
        near_b = rgb_b[:3, :, :].flatten()
    else:
        boundary_a = rgb_a.flatten()
        boundary_b = rgb_b.flatten()
        near_a = rgb_a.flatten()
        near_b = rgb_b.flatten()
    
    # 1. SSD - Sum of Squared Differences
    ssd = np.mean((boundary_a - boundary_b) ** 2)
    
    # 2. NCC - Normalized Cross-Correlation
    mean_a, mean_b = np.mean(boundary_a), np.mean(boundary_b)
    std_a, std_b = np.std(boundary_a) + 1e-10, np.std(boundary_b) + 1e-10
    ncc = np.mean((boundary_a - mean_a) * (boundary_b - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100  # Same scale as legacy
    
    # 3. Continuity - seam difference relative to internal variance
    var_a = np.var(near_a) + 1e-10
    var_b = np.var(near_b) + 1e-10
    seam_diff = np.mean((boundary_a - boundary_b) ** 2)
    continuity = seam_diff / ((var_a + var_b) / 2)
    
    # EXACT legacy formula
    return 0.3 * np.sqrt(ssd) + 0.3 * ncc_cost + 0.4 * continuity * 10


def compute_gradient_cost(strips_a: dict, strips_b: dict, edge_a: str, edge_b: str) -> float:
    """
    Compute gradient continuity cost (SECONDARY FEATURE).
    
    Measures smoothness of intensity transitions across seam.
    """
    gray_a = strips_a['gray']
    gray_b = strips_b['gray']
    
    # Compute gradients
    if edge_a == 'right' and edge_b == 'left':
        # Horizontal gradient at boundary
        grad_a = gray_a[:, -1] - gray_a[:, -2] if gray_a.shape[1] > 1 else np.zeros_like(gray_a[:, -1])
        grad_b = gray_b[:, 1] - gray_b[:, 0] if gray_b.shape[1] > 1 else np.zeros_like(gray_b[:, 0])
    elif edge_a == 'bottom' and edge_b == 'top':
        # Vertical gradient at boundary
        grad_a = gray_a[-1, :] - gray_a[-2, :] if gray_a.shape[0] > 1 else np.zeros_like(gray_a[-1, :])
        grad_b = gray_b[1, :] - gray_b[0, :] if gray_b.shape[0] > 1 else np.zeros_like(gray_b[0, :])
    else:
        return 0.0
    
    # Gradient should be similar (smooth transition)
    grad_diff = np.mean((grad_a.flatten() - grad_b.flatten()) ** 2)
    
    return np.sqrt(grad_diff)


def compute_vertical_asymmetry_penalty(strips_a: dict, strips_b: dict, 
                                        edge_a: str, edge_b: str, N: int = 3) -> float:
    """
    Compute directional asymmetry penalty for VERTICAL seams only.
    
    For A (bottom) → B (top) placement:
    - Compute the gradient trend WITHIN A approaching the seam
    - Compute the gradient ACROSS the seam (A's bottom row → B's top row)
    - Compute the gradient trend WITHIN B leaving the seam
    - Penalize if the cross-seam gradient doesn't match the internal trends
    
    This is ASYMMETRIC: cost(A→B) ≠ cost(B→A) because the seam values differ.
    
    Args:
        strips_a, strips_b: edge strips with 'gray' channel
        edge_a, edge_b: edge identifiers
        N: number of rows to compute gradient over
        
    Returns:
        Penalty (0 if horizontal seam, positive if vertical with gradient mismatch)
    """
    # Only apply to vertical seams (A bottom → B top)
    if not (edge_a == 'bottom' and edge_b == 'top'):
        return 0.0
    
    gray_a = strips_a['gray']  # Shape: (strip_width, W)
    gray_b = strips_b['gray']
    
    h_a = gray_a.shape[0]
    h_b = gray_b.shape[0]
    
    # Get boundary values
    boundary_a = np.mean(gray_a[-1, :])  # Bottom row of A
    boundary_b = np.mean(gray_b[0, :])   # Top row of B
    
    # Gradient ACROSS the seam
    grad_seam = boundary_b - boundary_a
    
    # Gradient trend WITHIN A (approaching seam from above)
    # Positive = getting brighter going down
    if h_a >= N + 1:
        upper_a = np.mean(gray_a[-N-1, :])
        lower_a = np.mean(gray_a[-1, :])
        grad_a = (lower_a - upper_a) / N
    else:
        grad_a = (np.mean(gray_a[-1, :]) - np.mean(gray_a[0, :])) / max(h_a - 1, 1)
    
    # Gradient trend WITHIN B (leaving seam going down)
    if h_b >= N + 1:
        upper_b = np.mean(gray_b[0, :])
        lower_b = np.mean(gray_b[N, :])
        grad_b = (lower_b - upper_b) / N
    else:
        grad_b = (np.mean(gray_b[-1, :]) - np.mean(gray_b[0, :])) / max(h_b - 1, 1)
    
    # Predicted seam gradient based on internal trends
    # If both pieces have consistent gradient, seam should follow
    predicted_seam_grad = (grad_a + grad_b) / 2
    
    # Penalty: how much does actual seam gradient deviate from prediction?
    # This is ASYMMETRIC because boundary_a and boundary_b values differ when pieces swap
    seam_deviation = abs(grad_seam - predicted_seam_grad)
    
    # Also penalize sign reversal between internal gradient and seam gradient
    sign_penalty = 0.0
    avg_internal_grad = (grad_a + grad_b) / 2
    if avg_internal_grad * grad_seam < 0 and abs(avg_internal_grad) > 1.0:
        # Internal trend says one direction, seam goes opposite
        sign_penalty = abs(grad_seam) * 0.3
    
    # Scale to be a weak tie-breaker (not dominant)
    return seam_deviation * 0.1 + sign_penalty


def seam_cost(artifacts: Dict[int, dict], A: int, edge_a: str, B: int, edge_b: str,
              weights: SeamCostWeights = None, strip_width: int = 10) -> float:
    """
    Compute multi-feature seam cost between two piece edges.
    
    REQUIRES Phase 1 artifacts - will fail without them.
    
    Args:
        artifacts: dict of piece_id -> artifact dict (MANDATORY)
        A: piece id for first piece
        edge_a: edge of piece A ('top', 'bottom', 'left', 'right')
        B: piece id for second piece  
        edge_b: edge of piece B
        weights: feature weights (default: artifact-first)
        strip_width: pixels to extract from edge
    
    Returns:
        Combined score (lower is better match)
    
    Raises:
        ValueError: If artifacts are missing or invalid
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # Validate artifacts exist (will raise if missing)
    validate_artifact(artifacts[A], A)
    validate_artifact(artifacts[B], B)
    
    # Extract strips from all channels
    strips_a = extract_edge_strip(artifacts[A], edge_a, strip_width)
    strips_b = extract_edge_strip(artifacts[B], edge_b, strip_width)
    
    # PRIMARY: RGB cost (legacy algorithm - main ranking signal)
    rgb_cost = compute_rgb_cost_legacy(strips_a, strips_b, edge_a, edge_b)
    
    # REGULARIZERS: Artifact-based penalties
    lowfreq_cost = compute_lowfreq_cost(strips_a, strips_b, edge_a, edge_b)
    edge_cost = compute_edge_cost(strips_a, strips_b, edge_a, edge_b)
    texture_cost = compute_texture_cost(strips_a, strips_b, edge_a, edge_b)
    
    # DIRECTIONAL ASYMMETRY: Vertical seam gradient penalty (tie-breaker)
    # This breaks symmetry between (A above B) vs (B above A)
    vertical_asymmetry = compute_vertical_asymmetry_penalty(strips_a, strips_b, edge_a, edge_b)
    
    # Hierarchical combination: RGB primary + artifact regularization + asymmetry
    total = (
        weights.rgb_weight * rgb_cost +
        weights.lowfreq_weight * lowfreq_cost +
        weights.edge_weight * edge_cost +
        weights.texture_weight * texture_cost +
        vertical_asymmetry  # Weak additive penalty (not weighted)
    )
    
    return total


def seam_cost_rgb_only_from_artifacts(artifacts: Dict[int, dict], A: int, edge_a: str, 
                                       B: int, edge_b: str, strip_width: int = 10,
                                       include_asymmetry: bool = True) -> float:
    """
    Compute RGB-only seam cost from artifacts (for beam search ranking).
    
    Uses EXACT legacy algorithm. This is the PRIMARY ranking signal.
    Optionally includes directional asymmetry penalty for vertical seams.
    """
    strips_a = extract_edge_strip(artifacts[A], edge_a, strip_width)
    strips_b = extract_edge_strip(artifacts[B], edge_b, strip_width)
    
    rgb_cost = compute_rgb_cost_legacy(strips_a, strips_b, edge_a, edge_b)
    
    if include_asymmetry:
        # Add directional asymmetry penalty for vertical seams
        asymmetry = compute_vertical_asymmetry_penalty(strips_a, strips_b, edge_a, edge_b)
        return rgb_cost + asymmetry
    
    return rgb_cost


def seam_cost_with_regularization(artifacts: Dict[int, dict], A: int, edge_a: str,
                                   B: int, edge_b: str, weights: SeamCostWeights = None,
                                   strip_width: int = 10) -> float:
    """
    Compute seam cost with artifact regularization (for refinement).
    
    RGB is PRIMARY, artifacts REGULARIZE.
    """
    return seam_cost(artifacts, A, edge_a, B, edge_b, weights, strip_width)


def build_match_table_rgb_only(artifacts: Dict[int, dict]) -> dict:
    """
    Build match table using RGB-only scoring (legacy behavior).
    
    Use this for beam search ranking to match baseline.
    """
    validate_artifacts(artifacts)
    
    piece_ids = list(artifacts.keys())
    match = {a: {b: {} for b in piece_ids} for a in piece_ids}
    
    for A in piece_ids:
        for B in piece_ids:
            if A == B:
                match[A][B]['right'] = float('inf')
                match[A][B]['bottom'] = float('inf')
            else:
                match[A][B]['right'] = seam_cost_rgb_only_from_artifacts(artifacts, A, 'right', B, 'left')
                match[A][B]['bottom'] = seam_cost_rgb_only_from_artifacts(artifacts, A, 'bottom', B, 'top')
    
    return match


def build_match_table(artifacts: Dict[int, dict], weights: SeamCostWeights = None) -> dict:
    """
    Precompute all pairwise edge matching costs using artifacts.
    
    Args:
        artifacts: dict of piece_id -> artifact dict (MANDATORY)
        weights: feature weights
    
    Returns:
        match: dict where match[A][B]['right'] = cost of placing B right of A
    """
    # Validate all artifacts first
    validate_artifacts(artifacts)
    
    piece_ids = list(artifacts.keys())
    match = {a: {b: {} for b in piece_ids} for a in piece_ids}
    
    for A in piece_ids:
        for B in piece_ids:
            if A == B:
                match[A][B]['right'] = float('inf')
                match[A][B]['bottom'] = float('inf')
            else:
                match[A][B]['right'] = seam_cost(artifacts, A, 'right', B, 'left', weights)
                match[A][B]['bottom'] = seam_cost(artifacts, A, 'bottom', B, 'top', weights)
    
    return match


def score_board(match: dict, board: dict, grid_size: int) -> float:
    """
    Compute total puzzle score for a board arrangement.
    
    Args:
        match: precomputed match table
        board: dict mapping (r, c) -> piece_id
        grid_size: size of grid
    
    Returns:
        Total seam cost (lower is better)
    """
    score = 0.0
    
    # Horizontal seams
    for r in range(grid_size):
        for c in range(grid_size - 1):
            A = board[(r, c)]
            B = board[(r, c + 1)]
            score += match[A][B]['right']
    
    # Vertical seams
    for r in range(grid_size - 1):
        for c in range(grid_size):
            A = board[(r, c)]
            B = board[(r + 1, c)]
            score += match[A][B]['bottom']
    
    return score


# =============================================================================
# LEGACY RGB-ONLY FUNCTIONS (for backward compatibility / baseline comparison)
# =============================================================================

def seam_cost_rgb_only(pieces: Dict[int, np.ndarray], A: int, edge_a: str, 
                       B: int, edge_b: str) -> float:
    """
    LEGACY/BASELINE: RGB-only seam cost (no artifacts).
    
    This is the original algorithm - kept for comparison purposes.
    New code should use seam_cost() with artifacts.
    """
    import warnings
    warnings.warn(
        "seam_cost_rgb_only is a legacy baseline. Use seam_cost() with artifacts for better results.",
        DeprecationWarning
    )
    
    def extract_edge_rgb(piece, edge, strip_width=10):
        if edge == 'top':
            return piece[:strip_width, :, :].astype(np.float32)
        elif edge == 'bottom':
            return piece[-strip_width:, :, :].astype(np.float32)
        elif edge == 'left':
            return piece[:, :strip_width, :].astype(np.float32)
        elif edge == 'right':
            return piece[:, -strip_width:, :].astype(np.float32)
    
    strip_a = extract_edge_rgb(pieces[A], edge_a)
    strip_b = extract_edge_rgb(pieces[B], edge_b)
    
    if edge_a == 'right' and edge_b == 'left':
        pixels_a = strip_a[:, -1, :].flatten()
        pixels_b = strip_b[:, 0, :].flatten()
        near_a = strip_a[:, -3:, :].flatten()
        near_b = strip_b[:, :3, :].flatten()
    elif edge_a == 'bottom' and edge_b == 'top':
        pixels_a = strip_a[-1, :, :].flatten()
        pixels_b = strip_b[0, :, :].flatten()
        near_a = strip_a[-3:, :, :].flatten()
        near_b = strip_b[:3, :, :].flatten()
    else:
        pixels_a = strip_a.flatten()
        pixels_b = strip_b.flatten()
        near_a = strip_a.flatten()
        near_b = strip_b.flatten()
    
    ssd = np.mean((pixels_a - pixels_b) ** 2)
    
    mean_a, mean_b = np.mean(pixels_a), np.mean(pixels_b)
    std_a, std_b = np.std(pixels_a) + 1e-10, np.std(pixels_b) + 1e-10
    ncc = np.mean((pixels_a - mean_a) * (pixels_b - mean_b)) / (std_a * std_b)
    ncc_cost = (1.0 - ncc) * 100
    
    var_a = np.var(near_a) + 1e-10
    var_b = np.var(near_b) + 1e-10
    seam_diff = np.mean((pixels_a - pixels_b) ** 2)
    continuity = seam_diff / ((var_a + var_b) / 2)
    
    return 0.3 * np.sqrt(ssd) + 0.3 * ncc_cost + 0.4 * continuity * 10
