"""
4x4 Puzzle Solver - Hierarchical Scoring

ARCHITECTURE:
- PRIMARY: RGB continuity (legacy algorithm) for beam search ranking
- REGULARIZERS: Artifacts (edges, gray, blur) for refinement penalties
- GLOBAL: Consistency penalties only on complete boards

Scoring hierarchy:
1. Beam search: RGB-only match table (matches legacy baseline)
2. Candidate re-ranking: RGB + artifact regularization
3. Refinement: Full score with global consistency

Artifacts are MANDATORY but REGULARIZE RGB, not override it.
"""

import random
import numpy as np
from typing import Dict, Tuple, List, Set
from collections import defaultdict

from .seam_cost import (
    build_match_table, build_match_table_rgb_only, score_board, 
    validate_artifacts, SeamCostWeights
)


def count_constraints(board: dict, pos: Tuple[int, int], grid_size: int) -> int:
    """Count how many neighbor constraints apply at a position."""
    r, c = pos
    count = 0
    if c > 0 and (r, c - 1) in board:
        count += 1
    if r > 0 and (r - 1, c) in board:
        count += 1
    return count


def compute_placement_cost(match: dict, board: dict, pid: int, 
                           r: int, c: int) -> Tuple[float, int]:
    """
    Compute placement cost and constraint count.
    
    Returns:
        (cost, num_constraints)
    """
    cost = 0.0
    constraints = 0
    
    if c > 0 and (r, c - 1) in board:
        left_pid = board[(r, c - 1)]
        cost += match[left_pid][pid]['right']
        constraints += 1
    
    if r > 0 and (r - 1, c) in board:
        top_pid = board[(r - 1, c)]
        cost += match[top_pid][pid]['bottom']
        constraints += 1
    
    return cost, constraints


def compute_lookahead_penalty(match: dict, board: dict, pid: int,
                               r: int, c: int, used: Set[int], 
                               grid_size: int) -> float:
    """
    Compute penalty based on how well remaining pieces can fill adjacent positions.
    
    Penalizes placements that leave hard-to-fill neighbors.
    """
    penalty = 0.0
    remaining = set(range(grid_size * grid_size)) - used - {pid}
    
    # Check right neighbor (if empty and within bounds)
    if c + 1 < grid_size and (r, c + 1) not in board:
        # Find best match for right position
        best_right = float('inf')
        for other_pid in remaining:
            cost = match[pid][other_pid]['right']
            best_right = min(best_right, cost)
        if best_right < float('inf'):
            penalty += best_right * 0.1  # Small weight
    
    # Check bottom neighbor
    if r + 1 < grid_size and (r + 1, c) not in board:
        best_bottom = float('inf')
        for other_pid in remaining:
            cost = match[pid][other_pid]['bottom']
            best_bottom = min(best_bottom, cost)
        if best_bottom < float('inf'):
            penalty += best_bottom * 0.1
    
    return penalty


def get_first_row_signature(board: dict, grid_size: int) -> tuple:
    """Get signature of first row for diversity tracking."""
    return tuple(board.get((0, c), -1) for c in range(grid_size))


# =============================================================================
# BORDER CONSISTENCY CONSTRAINT (fixes background sliding)
# =============================================================================

def compute_edge_border_likelihood(artifact: dict, edge: str) -> float:
    """
    Estimate likelihood that an edge is an outer border.
    
    Outer borders typically have:
    - Low variance (uniform background)
    - Low edge density (few Canny edges)
    - Low texture entropy (smooth/flat)
    
    Args:
        artifact: piece artifact with 'gray', 'edges', 'rgb'
        edge: 'top', 'bottom', 'left', 'right'
    
    Returns:
        Border likelihood score (higher = more likely to be outer border)
    """
    gray = artifact['gray'].astype(np.float32)
    edges = artifact['edges'].astype(np.float32)
    rgb = artifact['rgb'].astype(np.float32)
    
    # Extract edge strip (10 pixels wide)
    strip_width = 10
    if edge == 'top':
        gray_strip = gray[:strip_width, :]
        edge_strip = edges[:strip_width, :]
        rgb_strip = rgb[:strip_width, :, :]
    elif edge == 'bottom':
        gray_strip = gray[-strip_width:, :]
        edge_strip = edges[-strip_width:, :]
        rgb_strip = rgb[-strip_width:, :, :]
    elif edge == 'left':
        gray_strip = gray[:, :strip_width]
        edge_strip = edges[:, :strip_width]
        rgb_strip = rgb[:, :strip_width, :]
    elif edge == 'right':
        gray_strip = gray[:, -strip_width:]
        edge_strip = edges[:, -strip_width:]
        rgb_strip = rgb[:, -strip_width:, :]
    else:
        return 0.0
    
    # 1. Low variance indicator (uniform = likely border)
    variance = np.var(gray_strip)
    low_variance_score = max(0, 1.0 - variance / 500.0)  # Normalize, cap at 1
    
    # 2. Low edge density (few Canny edges = likely border)
    edge_density = np.mean(edge_strip > 0)
    low_edge_score = 1.0 - edge_density  # 0 edges = score 1
    
    # 3. Low texture entropy (smooth = likely border)
    # Use local standard deviation as proxy for texture
    local_std = np.std(gray_strip)
    low_texture_score = max(0, 1.0 - local_std / 30.0)
    
    # 4. Color uniformity (single color = likely border)
    color_variance = np.mean([np.var(rgb_strip[:, :, c]) for c in range(3)])
    low_color_var_score = max(0, 1.0 - color_variance / 500.0)
    
    # 5. Extreme intensity (very dark or very bright = likely border)
    mean_intensity = np.mean(gray_strip)
    extreme_intensity_score = 0.0
    if mean_intensity < 30 or mean_intensity > 225:
        extreme_intensity_score = 0.5
    
    # Combine scores (weighted average)
    border_likelihood = (
        0.30 * low_variance_score +
        0.25 * low_edge_score +
        0.20 * low_texture_score +
        0.15 * low_color_var_score +
        0.10 * extreme_intensity_score
    )
    
    return border_likelihood


def compute_piece_border_likelihoods(artifact: dict) -> dict:
    """
    Compute border likelihood for all four edges of a piece.
    
    Returns:
        dict with 'top', 'bottom', 'left', 'right' border likelihoods
    """
    return {
        'top': compute_edge_border_likelihood(artifact, 'top'),
        'bottom': compute_edge_border_likelihood(artifact, 'bottom'),
        'left': compute_edge_border_likelihood(artifact, 'left'),
        'right': compute_edge_border_likelihood(artifact, 'right')
    }


def compute_border_consistency_penalty(artifacts: Dict[int, dict], board: dict,
                                        grid_size: int,
                                        threshold: float = 0.6) -> float:
    """
    Penalize placing "outer-looking" edges inside the board.
    
    If an edge has high border likelihood but is placed as an internal seam,
    that's suspicious - it suggests the assembly might be shifted.
    
    Args:
        artifacts: piece artifacts
        board: current board arrangement
        grid_size: puzzle grid size
        threshold: border likelihood threshold to consider "outer-looking"
    
    Returns:
        Penalty score (higher = more border violations)
    """
    if len(board) < grid_size * grid_size:
        return 0.0  # Only evaluate complete boards
    
    penalty = 0.0
    
    # Precompute border likelihoods for all pieces
    border_scores = {}
    for pid in artifacts.keys():
        border_scores[pid] = compute_piece_border_likelihoods(artifacts[pid])
    
    # Check each internal seam
    for r in range(grid_size):
        for c in range(grid_size):
            pid = board[(r, c)]
            scores = border_scores[pid]
            
            # Check RIGHT edge (internal if c < grid_size - 1)
            if c < grid_size - 1:
                # This is an internal horizontal seam
                right_border_score = scores['right']
                if right_border_score > threshold:
                    # High border likelihood on internal edge = penalty
                    penalty += (right_border_score - threshold) * 2.0
                
                # Also check the left edge of the right neighbor
                right_neighbor_pid = board[(r, c + 1)]
                left_border_score = border_scores[right_neighbor_pid]['left']
                if left_border_score > threshold:
                    penalty += (left_border_score - threshold) * 2.0
            
            # Check BOTTOM edge (internal if r < grid_size - 1)
            if r < grid_size - 1:
                # This is an internal vertical seam
                bottom_border_score = scores['bottom']
                if bottom_border_score > threshold:
                    penalty += (bottom_border_score - threshold) * 2.0
                
                # Also check the top edge of the bottom neighbor
                bottom_neighbor_pid = board[(r + 1, c)]
                top_border_score = border_scores[bottom_neighbor_pid]['top']
                if top_border_score > threshold:
                    penalty += (top_border_score - threshold) * 2.0
    
    # Bonus: Check that actual border edges DO have high border likelihood
    # This rewards correct placement of border pieces
    border_bonus = 0.0
    
    # Top row - top edges should be borders
    for c in range(grid_size):
        pid = board[(0, c)]
        if border_scores[pid]['top'] > threshold:
            border_bonus -= 0.5  # Reward (negative penalty)
    
    # Bottom row - bottom edges should be borders
    for c in range(grid_size):
        pid = board[(grid_size - 1, c)]
        if border_scores[pid]['bottom'] > threshold:
            border_bonus -= 0.5
    
    # Left column - left edges should be borders
    for r in range(grid_size):
        pid = board[(r, 0)]
        if border_scores[pid]['left'] > threshold:
            border_bonus -= 0.5
    
    # Right column - right edges should be borders
    for r in range(grid_size):
        pid = board[(r, grid_size - 1)]
        if border_scores[pid]['right'] > threshold:
            border_bonus -= 0.5
    
    return penalty + border_bonus


# =============================================================================
# ABSOLUTE POSITION PRIOR (breaks mirror/shift ambiguity)
# =============================================================================

def compute_piece_position_features(artifact: dict) -> dict:
    """
    Extract features that hint at absolute position in original image.
    
    Uses asymmetric features that differ between left/right, top/bottom:
    - Corner intensities (corners are unique anchors)
    - Edge-to-center gradients (pieces near edges have different gradients)
    - Asymmetric intensity moments
    """
    gray = artifact['gray'].astype(np.float32)
    h, w = gray.shape
    
    # Corner intensities (5x5 patches)
    tl = np.mean(gray[:5, :5])      # top-left
    tr = np.mean(gray[:5, -5:])     # top-right
    bl = np.mean(gray[-5:, :5])     # bottom-left
    br = np.mean(gray[-5:, -5:])    # bottom-right
    
    # Edge intensities
    top = np.mean(gray[:5, :])
    bottom = np.mean(gray[-5:, :])
    left = np.mean(gray[:, :5])
    right = np.mean(gray[:, -5:])
    center = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
    
    # Asymmetry measures
    h_asym = right - left           # positive = brighter on right
    v_asym = bottom - top           # positive = brighter on bottom
    diag_asym = (tr + bl) - (tl + br)  # diagonal asymmetry
    
    return {
        'corners': (tl, tr, bl, br),
        'edges': (top, bottom, left, right),
        'center': center,
        'h_asym': h_asym,
        'v_asym': v_asym,
        'diag_asym': diag_asym,
        'mean': np.mean(gray)
    }


def compute_position_prior_penalty(artifacts: Dict[int, dict], board: dict,
                                    grid_size: int) -> float:
    """
    Compute weak absolute position prior penalty.
    
    This breaks mirror/shift ambiguity by checking if the global
    intensity gradient of the assembled image is "natural":
    - Images typically have consistent lighting direction
    - Corner pieces should have corner-like features
    - Edge pieces should have edge-like gradients
    
    The penalty is VERY WEAK - only breaks ties between otherwise equal solutions.
    
    Args:
        artifacts: piece artifacts
        board: current board arrangement
        grid_size: puzzle grid size
    
    Returns:
        Penalty score (higher = less likely to be correct orientation)
    """
    if len(board) < grid_size * grid_size:
        return 0.0  # Only evaluate complete boards
    
    penalty = 0.0
    
    # Extract position features for all pieces
    features = {}
    for pos, pid in board.items():
        features[pos] = compute_piece_position_features(artifacts[pid])
    
    # 1. Global gradient consistency
    # Compute overall left-to-right and top-to-bottom gradients
    left_col_mean = np.mean([features[(r, 0)]['mean'] for r in range(grid_size)])
    right_col_mean = np.mean([features[(r, grid_size-1)]['mean'] for r in range(grid_size)])
    top_row_mean = np.mean([features[(0, c)]['mean'] for c in range(grid_size)])
    bottom_row_mean = np.mean([features[(grid_size-1, c)]['mean'] for c in range(grid_size)])
    
    global_h_grad = right_col_mean - left_col_mean
    global_v_grad = bottom_row_mean - top_row_mean
    
    # 2. Check if piece asymmetries align with their positions
    # Pieces on the left should have h_asym suggesting they're on the left, etc.
    for r in range(grid_size):
        for c in range(grid_size):
            feat = features[(r, c)]
            
            # Horizontal position consistency
            # Pieces on left (c < grid_size/2) should have features consistent with left side
            expected_h_bias = (c - grid_size/2 + 0.5) / grid_size  # -0.375 to 0.375 for 4x4
            actual_h_bias = feat['h_asym'] / 50.0  # Normalize
            
            # Vertical position consistency
            expected_v_bias = (r - grid_size/2 + 0.5) / grid_size
            actual_v_bias = feat['v_asym'] / 50.0
            
            # Penalize if piece asymmetry contradicts expected position
            # This is VERY weak - just a tie-breaker
            h_mismatch = abs(expected_h_bias - actual_h_bias) * 0.5
            v_mismatch = abs(expected_v_bias - actual_v_bias) * 0.5
            
            penalty += h_mismatch + v_mismatch
    
    # 3. Corner piece validation
    # True corner pieces often have distinctive corner features
    corners = [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]
    corner_names = ['tl', 'tr', 'bl', 'br']
    
    for (r, c), name in zip(corners, corner_names):
        feat = features[(r, c)]
        tl, tr, bl, br = feat['corners']
        
        # The corner of the piece that's at the image corner should be distinctive
        if name == 'tl':
            # Top-left piece: its top-left corner should be an extremum
            corner_val = tl
        elif name == 'tr':
            corner_val = tr
        elif name == 'bl':
            corner_val = bl
        else:
            corner_val = br
        
        # Weak penalty if corner isn't distinctive (close to center intensity)
        center = feat['center']
        corner_distinctiveness = abs(corner_val - center)
        if corner_distinctiveness < 10:  # Not very distinctive
            penalty += 0.5
    
    # 4. Edge piece gradient alignment
    # Pieces on edges should have gradients pointing inward
    for c in range(grid_size):
        # Top row: pieces should have gradients pointing down (into image)
        top_feat = features[(0, c)]
        if top_feat['v_asym'] < -5:  # Gradient pointing up (away from image)
            penalty += 0.3
        
        # Bottom row: pieces should have gradients pointing up
        bottom_feat = features[(grid_size-1, c)]
        if bottom_feat['v_asym'] > 5:
            penalty += 0.3
    
    for r in range(grid_size):
        # Left column: pieces should have gradients pointing right
        left_feat = features[(r, 0)]
        if left_feat['h_asym'] < -5:
            penalty += 0.3
        
        # Right column: pieces should have gradients pointing left
        right_feat = features[(r, grid_size-1)]
        if right_feat['h_asym'] > 5:
            penalty += 0.3
    
    return penalty


def score_board_with_position_prior(artifacts: Dict[int, dict], match: dict,
                                     board: dict, grid_size: int,
                                     prior_weight: float = 0.05) -> float:
    """
    Score board with weak position prior to break orientation ambiguity.
    
    Args:
        artifacts: piece artifacts
        match: match table
        board: board arrangement
        grid_size: puzzle grid size
        prior_weight: weight for position prior (should be VERY small)
    
    Returns:
        Combined score
    """
    local_score = score_board(match, board, grid_size)
    position_penalty = compute_position_prior_penalty(artifacts, board, grid_size)
    
    return local_score + prior_weight * position_penalty


# =============================================================================
# GLOBAL CONSISTENCY PENALTIES
# =============================================================================

def compute_row_gradient_penalty(artifacts: Dict[int, dict], board: dict, 
                                  grid_size: int) -> float:
    """
    Penalize rows with inconsistent horizontal gradient direction.
    
    For each row, compute the average horizontal gradient (left-to-right intensity change).
    Penalize if adjacent pieces have drastically different gradient directions.
    """
    penalty = 0.0
    
    for r in range(grid_size):
        row_gradients = []
        for c in range(grid_size):
            if (r, c) not in board:
                continue
            pid = board[(r, c)]
            gray = artifacts[pid]['gray'].astype(np.float32)
            
            # Horizontal gradient: right side minus left side
            h_grad = np.mean(gray[:, -5:]) - np.mean(gray[:, :5])
            row_gradients.append(h_grad)
        
        if len(row_gradients) < 2:
            continue
        
        # Penalize sign changes in gradient direction within a row
        for i in range(len(row_gradients) - 1):
            g1, g2 = row_gradients[i], row_gradients[i + 1]
            # Large sign flip = inconsistent gradient flow
            if g1 * g2 < 0 and abs(g1) > 5 and abs(g2) > 5:
                penalty += abs(g1 - g2) * 0.1
    
    return penalty


def compute_col_gradient_penalty(artifacts: Dict[int, dict], board: dict,
                                  grid_size: int) -> float:
    """
    Penalize columns with inconsistent vertical gradient direction.
    
    For each column, compute the average vertical gradient (top-to-bottom intensity change).
    Penalize if adjacent pieces have drastically different gradient directions.
    """
    penalty = 0.0
    
    for c in range(grid_size):
        col_gradients = []
        for r in range(grid_size):
            if (r, c) not in board:
                continue
            pid = board[(r, c)]
            gray = artifacts[pid]['gray'].astype(np.float32)
            
            # Vertical gradient: bottom side minus top side
            v_grad = np.mean(gray[-5:, :]) - np.mean(gray[:5, :])
            col_gradients.append(v_grad)
        
        if len(col_gradients) < 2:
            continue
        
        # Penalize sign changes in gradient direction within a column
        for i in range(len(col_gradients) - 1):
            g1, g2 = col_gradients[i], col_gradients[i + 1]
            if g1 * g2 < 0 and abs(g1) > 5 and abs(g2) > 5:
                penalty += abs(g1 - g2) * 0.1
    
    return penalty


def compute_texture_variance_penalty(artifacts: Dict[int, dict], board: dict,
                                      grid_size: int) -> float:
    """
    Penalize arrangements where texture variance changes abruptly.
    
    Smooth images should have smooth variance transitions.
    Textured regions should be grouped together.
    """
    penalty = 0.0
    
    # Compute variance for each placed piece
    variances = {}
    for pos, pid in board.items():
        gray = artifacts[pid]['gray'].astype(np.float32)
        variances[pos] = np.var(gray)
    
    # Penalize large variance jumps between adjacent pieces
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in variances:
                continue
            
            v1 = variances[(r, c)]
            
            # Check right neighbor
            if (r, c + 1) in variances:
                v2 = variances[(r, c + 1)]
                ratio = max(v1, v2) / (min(v1, v2) + 1e-10)
                if ratio > 3.0:  # More than 3x variance difference
                    penalty += (ratio - 3.0) * 2.0
            
            # Check bottom neighbor
            if (r + 1, c) in variances:
                v2 = variances[(r + 1, c)]
                ratio = max(v1, v2) / (min(v1, v2) + 1e-10)
                if ratio > 3.0:
                    penalty += (ratio - 3.0) * 2.0
    
    return penalty


def compute_intensity_flow_penalty(artifacts: Dict[int, dict], board: dict,
                                    grid_size: int) -> float:
    """
    Penalize arrangements that break natural intensity flow.
    
    Checks if the overall intensity pattern (bright/dark regions) is coherent.
    """
    penalty = 0.0
    
    # Build intensity map
    intensities = {}
    for pos, pid in board.items():
        gray = artifacts[pid]['gray'].astype(np.float32)
        intensities[pos] = np.mean(gray)
    
    if len(intensities) < grid_size * grid_size:
        return 0.0  # Only evaluate on complete boards
    
    # Check row-wise intensity coherence
    for r in range(grid_size):
        row_intensities = [intensities[(r, c)] for c in range(grid_size)]
        # Compute second derivative (acceleration of intensity change)
        for i in range(len(row_intensities) - 2):
            d1 = row_intensities[i + 1] - row_intensities[i]
            d2 = row_intensities[i + 2] - row_intensities[i + 1]
            # Large acceleration = abrupt change in intensity trend
            accel = abs(d2 - d1)
            if accel > 30:  # Threshold for "abrupt"
                penalty += (accel - 30) * 0.05
    
    # Check column-wise intensity coherence
    for c in range(grid_size):
        col_intensities = [intensities[(r, c)] for r in range(grid_size)]
        for i in range(len(col_intensities) - 2):
            d1 = col_intensities[i + 1] - col_intensities[i]
            d2 = col_intensities[i + 2] - col_intensities[i + 1]
            accel = abs(d2 - d1)
            if accel > 30:
                penalty += (accel - 30) * 0.05
    
    return penalty


def compute_global_consistency_penalty(artifacts: Dict[int, dict], board: dict,
                                        grid_size: int, 
                                        weights: dict = None) -> float:
    """
    Compute total global consistency penalty for a board.
    
    Only meaningful on near-complete or complete boards.
    
    Args:
        artifacts: piece artifacts
        board: current board state
        grid_size: puzzle grid size
        weights: optional dict with penalty weights
    
    Returns:
        Total global penalty (higher = worse global consistency)
    """
    if weights is None:
        weights = {
            'row_gradient': 1.0,
            'col_gradient': 1.0,
            'texture_variance': 0.5,
            'intensity_flow': 0.5
        }
    
    # Only apply on boards that are at least 75% complete
    completion = len(board) / (grid_size * grid_size)
    if completion < 0.75:
        return 0.0
    
    penalty = 0.0
    penalty += weights['row_gradient'] * compute_row_gradient_penalty(artifacts, board, grid_size)
    penalty += weights['col_gradient'] * compute_col_gradient_penalty(artifacts, board, grid_size)
    penalty += weights['texture_variance'] * compute_texture_variance_penalty(artifacts, board, grid_size)
    penalty += weights['intensity_flow'] * compute_intensity_flow_penalty(artifacts, board, grid_size)
    
    return penalty


def score_board_with_global(artifacts: Dict[int, dict], match: dict, 
                            board: dict, grid_size: int,
                            global_weight: float = 0.15,
                            prior_weight: float = 0.05,
                            border_weight: float = 0.10) -> float:
    """
    Score a board with local seam cost and late-stage penalties.
    
    Hierarchy (seams remain primary):
    1. Local seam cost (RGB + artifact regularization) - PRIMARY
    2. Border consistency (penalize outer-looking edges inside) - SECONDARY
    3. Global consistency (gradient/texture coherence) - TERTIARY
    4. Position prior (break mirror/shift ambiguity) - WEAK TIE-BREAKER
    
    Args:
        artifacts: piece artifacts
        match: precomputed match table
        board: board arrangement
        grid_size: puzzle grid size
        global_weight: weight for global consistency penalty
        prior_weight: weight for position prior
        border_weight: weight for border consistency penalty
    
    Returns:
        Combined score (lower is better)
    """
    local_score = score_board(match, board, grid_size)
    border_penalty = compute_border_consistency_penalty(artifacts, board, grid_size)
    global_penalty = compute_global_consistency_penalty(artifacts, board, grid_size)
    position_penalty = compute_position_prior_penalty(artifacts, board, grid_size)
    
    return (local_score + 
            border_weight * border_penalty +
            global_weight * global_penalty + 
            prior_weight * position_penalty)


def precompute_border_likelihoods(artifacts: Dict[int, dict]) -> Dict[int, dict]:
    """
    Pre-compute border likelihood for all edges of all pieces.
    
    Called ONCE before beam search to identify likely border edges.
    
    Returns:
        dict mapping piece_id -> {'top': score, 'bottom': score, 'left': score, 'right': score}
    """
    border_scores = {}
    for pid, artifact in artifacts.items():
        border_scores[pid] = compute_piece_border_likelihoods(artifact)
    return border_scores


def compute_early_border_penalty(border_scores: dict, board: dict, pid: int,
                                  r: int, c: int, grid_size: int,
                                  threshold: float = 0.4) -> float:
    """
    Compute border penalty for placing a piece at a position DURING beam search.
    
    This is a CONSTRAINT, not a feature. It prevents global shifts by:
    1. HEAVILY penalizing border-like edges placed on internal seams
    2. Rewarding border-like edges placed on actual puzzle boundaries
    
    Args:
        border_scores: precomputed border likelihoods for all pieces
        board: current partial board
        pid: piece being placed
        r, c: position being filled
        grid_size: puzzle grid size
        threshold: border likelihood threshold (lower = more sensitive)
    
    Returns:
        Penalty to add to placement cost (higher = worse)
    """
    penalty = 0.0
    scores = border_scores[pid]
    
    # Penalty multiplier - must be strong enough to override seam cost ties
    INTERNAL_PENALTY = 50.0   # Heavy penalty for border edge on internal seam
    BOUNDARY_REWARD = 20.0    # Reward for border edge on actual boundary
    
    # Check LEFT edge of this piece
    if c > 0:
        # This piece's left edge faces an internal seam (not puzzle border)
        left_border_score = scores['left']
        if left_border_score > threshold:
            # High border likelihood on internal edge = BAD
            penalty += (left_border_score - threshold) * INTERNAL_PENALTY
        
        # Also check right edge of left neighbor
        left_neighbor_pid = board[(r, c - 1)]
        right_border_score = border_scores[left_neighbor_pid]['right']
        if right_border_score > threshold:
            penalty += (right_border_score - threshold) * INTERNAL_PENALTY
    else:
        # c == 0: This piece's left edge IS on puzzle border
        # Reward if it has high border likelihood
        left_border_score = scores['left']
        if left_border_score > threshold:
            penalty -= (left_border_score - threshold) * BOUNDARY_REWARD
        else:
            # Penalize non-border edge on boundary (mild)
            penalty += (threshold - left_border_score) * 5.0
    
    # Check TOP edge of this piece
    if r > 0:
        # This piece's top edge faces an internal seam
        top_border_score = scores['top']
        if top_border_score > threshold:
            penalty += (top_border_score - threshold) * INTERNAL_PENALTY
        
        # Also check bottom edge of top neighbor
        top_neighbor_pid = board[(r - 1, c)]
        bottom_border_score = border_scores[top_neighbor_pid]['bottom']
        if bottom_border_score > threshold:
            penalty += (bottom_border_score - threshold) * INTERNAL_PENALTY
    else:
        # r == 0: This piece's top edge IS on puzzle border
        top_border_score = scores['top']
        if top_border_score > threshold:
            penalty -= (top_border_score - threshold) * BOUNDARY_REWARD
        else:
            penalty += (threshold - top_border_score) * 5.0
    
    # Check if this piece is on RIGHT border (c == grid_size - 1)
    if c == grid_size - 1:
        right_border_score = scores['right']
        if right_border_score > threshold:
            penalty -= (right_border_score - threshold) * BOUNDARY_REWARD
        else:
            penalty += (threshold - right_border_score) * 5.0
    
    # Check if this piece is on BOTTOM border (r == grid_size - 1)
    if r == grid_size - 1:
        bottom_border_score = scores['bottom']
        if bottom_border_score > threshold:
            penalty -= (bottom_border_score - threshold) * BOUNDARY_REWARD
        else:
            penalty += (threshold - bottom_border_score) * 5.0
    
    return penalty


def beam_solve_with_border_constraint(artifacts: Dict[int, dict], match: dict,
                                       border_scores: dict,
                                       beam_width: int = 20000, grid_size: int = 4,
                                       border_penalty_weight: float = 1.0) -> Tuple[dict, float]:
    """
    Beam search with EARLY border constraint to prevent global shifts.
    
    Key difference from legacy: border penalties are applied DURING beam expansion,
    not just at final evaluation. This prevents early commitment to shifted solutions.
    
    Args:
        artifacts: piece artifacts
        match: RGB-only match table (seam cost remains primary)
        border_scores: precomputed border likelihoods
        beam_width: number of partial solutions to keep
        grid_size: puzzle grid size
        border_penalty_weight: weight for border penalty (1.0 = equal to seam cost scale)
    
    Returns:
        best_board, best_score
    """
    piece_ids = list(artifacts.keys())
    
    # State: (board_dict, used_set, cumulative_score)
    initial_state = ({}, frozenset(), 0.0)
    beam = [initial_state]
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    for pos_idx, (r, c) in enumerate(positions):
        next_beam = []
        
        for board, used, cum_score in beam:
            for pid in piece_ids:
                if pid in used:
                    continue
                
                # PRIMARY: RGB seam cost (unchanged from legacy)
                seam_cost = 0.0
                if c > 0:
                    left_pid = board[(r, c - 1)]
                    seam_cost += match[left_pid][pid]['right']
                if r > 0:
                    top_pid = board[(r - 1, c)]
                    seam_cost += match[top_pid][pid]['bottom']
                
                # EARLY GLOBAL CONSTRAINT: Border penalty
                border_penalty = compute_early_border_penalty(
                    border_scores, board, pid, r, c, grid_size
                )
                
                # Combined placement cost (seam primary, border secondary)
                placement_cost = seam_cost + border_penalty_weight * border_penalty
                
                new_board = board.copy()
                new_board[(r, c)] = pid
                new_used = used | {pid}
                new_score = cum_score + placement_cost
                
                next_beam.append((new_board, new_used, new_score))
        
        # Sort by combined score (seam + border penalty)
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        if (pos_idx + 1) % grid_size == 0:
            row_num = (pos_idx + 1) // grid_size
            print(f"    Row {row_num}: {len(next_beam)} -> {len(beam)} states")
    
    best_board, _, best_score = beam[0]
    
    # Return RGB-only score for comparison (without border penalty)
    rgb_score = score_board(match, best_board, grid_size)
    return best_board, rgb_score


def beam_solve_legacy(artifacts: Dict[int, dict], match: dict,
                      beam_width: int = 20000, grid_size: int = 4) -> Tuple[dict, float]:
    """
    Beam search using EXACT legacy algorithm (RGB-only, cumulative scoring).
    
    This matches Final4x4.py behavior exactly for baseline comparison.
    No normalization, no lookahead, no diversity - pure legacy.
    
    Args:
        artifacts: dict of piece_id -> artifact dict
        match: RGB-only match table
        beam_width: number of partial solutions to keep
        grid_size: size of grid
    
    Returns:
        best_board: dict mapping (r, c) -> piece_id
        best_score: total seam cost
    """
    piece_ids = list(artifacts.keys())
    
    # State: (board_dict, used_set, cumulative_score)
    initial_state = ({}, frozenset(), 0.0)
    beam = [initial_state]
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    for pos_idx, (r, c) in enumerate(positions):
        next_beam = []
        
        for board, used, cum_score in beam:
            for pid in piece_ids:
                if pid in used:
                    continue
                
                # Compute placement cost (legacy: simple cumulative)
                placement_cost = 0.0
                
                if c > 0:
                    left_pid = board[(r, c - 1)]
                    placement_cost += match[left_pid][pid]['right']
                
                if r > 0:
                    top_pid = board[(r - 1, c)]
                    placement_cost += match[top_pid][pid]['bottom']
                
                new_board = board.copy()
                new_board[(r, c)] = pid
                new_used = used | {pid}
                new_score = cum_score + placement_cost
                
                next_beam.append((new_board, new_used, new_score))
        
        # Simple sort by cumulative score (legacy behavior)
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        if (pos_idx + 1) % grid_size == 0:
            row_num = (pos_idx + 1) // grid_size
            print(f"    Row {row_num}: {len(next_beam)} -> {len(beam)} states")
    
    best_board, _, best_score = beam[0]
    return best_board, best_score


def rerank_with_artifacts(artifacts: Dict[int, dict], match_rgb: dict, 
                          match_full: dict, board: dict, grid_size: int = 4,
                          verbose: bool = True) -> Tuple[dict, float]:
    """
    Re-rank the beam search result using artifact regularization.
    
    This is a light touch - only re-evaluate if artifacts suggest improvement.
    RGB remains primary, artifacts only regularize.
    
    Args:
        artifacts: piece artifacts
        match_rgb: RGB-only match table
        match_full: Full match table with artifact regularization
        board: best board from beam search
        grid_size: puzzle grid size
        verbose: print progress
    
    Returns:
        best_board, best_score
    """
    # Get RGB score of current solution
    rgb_score = score_board(match_rgb, board, grid_size)
    full_score = score_board(match_full, board, grid_size)
    
    if verbose:
        print(f"    Current RGB score: {rgb_score:.2f}")
        print(f"    Current regularized score: {full_score:.2f}")
    
    # Try a few local swaps guided by artifact penalties
    # Only accept if RGB score doesn't degrade significantly
    current = board.copy()
    current_rgb = rgb_score
    current_full = full_score
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    improvements = 0
    
    # Light refinement: only accept swaps that improve full score
    # without degrading RGB score by more than 5%
    for _ in range(1000):
        pos1, pos2 = random.sample(positions, 2)
        
        current[pos1], current[pos2] = current[pos2], current[pos1]
        new_rgb = score_board(match_rgb, current, grid_size)
        new_full = score_board(match_full, current, grid_size)
        
        # Accept if: full score improves AND RGB doesn't degrade much
        rgb_degradation = (new_rgb - current_rgb) / (current_rgb + 1e-10)
        
        if new_full < current_full and rgb_degradation < 0.05:
            current_rgb = new_rgb
            current_full = new_full
            improvements += 1
        else:
            current[pos1], current[pos2] = current[pos2], current[pos1]
    
    if verbose:
        print(f"    Artifact-guided improvements: {improvements}")
        print(f"    Final RGB score: {current_rgb:.2f}")
    
    return current, current_full


def swap_hillclimb(artifacts: Dict[int, dict], match: dict, board: dict,
                   max_iterations: int = 5000, grid_size: int = 4,
                   use_global: bool = True) -> Tuple[dict, float, int]:
    """Basic swap hillclimb refinement with optional global scoring."""
    current = board.copy()
    
    if use_global:
        current_score = score_board_with_global(artifacts, match, current, grid_size)
    else:
        current_score = score_board(match, current, grid_size)
    
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    improvements = 0
    for _ in range(max_iterations):
        pos1, pos2 = random.sample(positions, 2)
        
        current[pos1], current[pos2] = current[pos2], current[pos1]
        
        if use_global:
            new_score = score_board_with_global(artifacts, match, current, grid_size)
        else:
            new_score = score_board(match, current, grid_size)
        
        if new_score < current_score:
            current_score = new_score
            improvements += 1
        else:
            current[pos1], current[pos2] = current[pos2], current[pos1]
    
    return current, current_score, improvements


def row_swap_refinement(artifacts: Dict[int, dict], match: dict, board: dict, 
                        grid_size: int = 4, use_global: bool = True) -> Tuple[dict, float, int]:
    """Try swapping entire rows with optional global scoring."""
    current = board.copy()
    
    if use_global:
        current_score = score_board_with_global(artifacts, match, current, grid_size)
    else:
        current_score = score_board(match, current, grid_size)
    
    improvements = 0
    
    for r1 in range(grid_size):
        for r2 in range(r1 + 1, grid_size):
            # Swap rows r1 and r2
            test = current.copy()
            for c in range(grid_size):
                test[(r1, c)], test[(r2, c)] = test[(r2, c)], test[(r1, c)]
            
            if use_global:
                new_score = score_board_with_global(artifacts, match, test, grid_size)
            else:
                new_score = score_board(match, test, grid_size)
            
            if new_score < current_score:
                current = test
                current_score = new_score
                improvements += 1
    
    return current, current_score, improvements


def col_swap_refinement(artifacts: Dict[int, dict], match: dict, board: dict, 
                        grid_size: int = 4, use_global: bool = True) -> Tuple[dict, float, int]:
    """Try swapping entire columns with optional global scoring."""
    current = board.copy()
    
    if use_global:
        current_score = score_board_with_global(artifacts, match, current, grid_size)
    else:
        current_score = score_board(match, current, grid_size)
    
    improvements = 0
    
    for c1 in range(grid_size):
        for c2 in range(c1 + 1, grid_size):
            # Swap columns c1 and c2
            test = current.copy()
            for r in range(grid_size):
                test[(r, c1)], test[(r, c2)] = test[(r, c2)], test[(r, c1)]
            
            if use_global:
                new_score = score_board_with_global(artifacts, match, test, grid_size)
            else:
                new_score = score_board(match, test, grid_size)
            
            if new_score < current_score:
                current = test
                current_score = new_score
                improvements += 1
    
    return current, current_score, improvements


def block_swap_refinement(artifacts: Dict[int, dict], match: dict, board: dict, 
                          grid_size: int = 4, use_global: bool = True) -> Tuple[dict, float, int]:
    """Try swapping 2x2 blocks with optional global scoring."""
    current = board.copy()
    
    if use_global:
        current_score = score_board_with_global(artifacts, match, current, grid_size)
    else:
        current_score = score_board(match, current, grid_size)
    
    improvements = 0
    
    # Get all 2x2 block positions
    blocks = []
    for r in range(grid_size - 1):
        for c in range(grid_size - 1):
            blocks.append((r, c))
    
    for i, (r1, c1) in enumerate(blocks):
        for r2, c2 in blocks[i + 1:]:
            # Check if blocks don't overlap
            if abs(r1 - r2) < 2 and abs(c1 - c2) < 2:
                continue
            
            # Swap 2x2 blocks
            test = current.copy()
            for dr in range(2):
                for dc in range(2):
                    pos1 = (r1 + dr, c1 + dc)
                    pos2 = (r2 + dr, c2 + dc)
                    test[pos1], test[pos2] = test[pos2], test[pos1]
            
            if use_global:
                new_score = score_board_with_global(artifacts, match, test, grid_size)
            else:
                new_score = score_board(match, test, grid_size)
            
            if new_score < current_score:
                current = test
                current_score = new_score
                improvements += 1
    
    return current, current_score, improvements


def multi_strategy_refinement(artifacts: Dict[int, dict], match: dict, board: dict, 
                               grid_size: int = 4, verbose: bool = True,
                               use_global: bool = True) -> Tuple[dict, float]:
    """
    Apply multiple refinement strategies with global consistency scoring.
    
    Args:
        artifacts: piece artifacts (needed for global penalties)
        match: precomputed match table
        board: initial board arrangement
        grid_size: puzzle grid size
        verbose: print progress
        use_global: use global consistency penalties in scoring
    
    Returns:
        refined_board, final_score
    """
    current = board.copy()
    
    if use_global:
        current_score = score_board_with_global(artifacts, match, current, grid_size)
    else:
        current_score = score_board(match, current, grid_size)
    
    total_improvements = 0
    
    # Strategy 1: Row swaps
    current, current_score, impr = row_swap_refinement(artifacts, match, current, grid_size, use_global)
    total_improvements += impr
    if verbose and impr > 0:
        print(f"    Row swaps: {impr} improvements")
    
    # Strategy 2: Column swaps
    current, current_score, impr = col_swap_refinement(artifacts, match, current, grid_size, use_global)
    total_improvements += impr
    if verbose and impr > 0:
        print(f"    Column swaps: {impr} improvements")
    
    # Strategy 3: Block swaps
    current, current_score, impr = block_swap_refinement(artifacts, match, current, grid_size, use_global)
    total_improvements += impr
    if verbose and impr > 0:
        print(f"    Block swaps: {impr} improvements")
    
    # Strategy 4: Random swaps (main refinement)
    current, current_score, impr = swap_hillclimb(artifacts, match, current, 5000, grid_size, use_global)
    total_improvements += impr
    if verbose:
        print(f"    Random swaps: {impr} improvements")
    
    # Strategy 5: Another round of structured swaps
    current, current_score, impr = row_swap_refinement(artifacts, match, current, grid_size, use_global)
    total_improvements += impr
    current, current_score, impr = col_swap_refinement(artifacts, match, current, grid_size, use_global)
    total_improvements += impr
    
    if verbose:
        print(f"    Total refinement improvements: {total_improvements}")
    
    return current, current_score


def solve_4x4(artifacts: Dict[int, dict], verbose: bool = True,
              weights: SeamCostWeights = None) -> Tuple[dict, list, float]:
    """
    Main solver function for 4x4 puzzles.
    
    HIERARCHICAL SCORING:
    1. Beam search uses RGB-only (matches legacy baseline)
    2. Re-ranking uses RGB + artifact regularization
    3. Refinement uses full score + global consistency
    
    Args:
        artifacts: dict mapping piece_id (0-15) to artifact dict
                   Each artifact MUST contain: 'rgb', 'gray', 'edges', 'blur'
        verbose: print progress info
        weights: optional custom feature weights
    
    Returns:
        board: dict mapping (row, col) -> piece_id
        arrangement: flat list of piece ids in row-major order
        score: final puzzle score
    """
    validate_artifacts(artifacts)
    
    if verbose:
        print("=" * 60)
        print("4x4 Puzzle Solver (Hierarchical: RGB Primary)")
        print("=" * 60)
        print(f"Pieces: {len(artifacts)}")
        print(f"Hierarchy: RGB (ranking) -> Artifacts (regularization) -> Global (final)")
    
    # Phase 1: Build RGB-only match table + precompute border likelihoods
    if verbose:
        print("\n[Phase 1] Building match tables and border analysis...")
    
    match_rgb = build_match_table_rgb_only(artifacts)
    border_scores = precompute_border_likelihoods(artifacts)
    
    if verbose:
        print("  RGB match table computed (16x16x2 = 512 scores)")
        # Show border analysis summary
        high_border_count = sum(
            1 for pid in border_scores 
            for edge in ['top', 'bottom', 'left', 'right']
            if border_scores[pid][edge] > 0.5
        )
        print(f"  Border analysis: {high_border_count} high-likelihood border edges detected")
    
    # Phase 2: Beam search with EARLY border constraint
    if verbose:
        print("\n[Phase 2] Beam search with early border constraint...")
    
    board, score = beam_solve_with_border_constraint(
        artifacts, match_rgb, border_scores,
        beam_width=20000, 
        grid_size=4,
        border_penalty_weight=1.0
    )
    
    if verbose:
        print(f"  Beam search RGB score: {score:.4f}")
    
    # Phase 3: Re-rank with artifact regularization
    if verbose:
        print("\n[Phase 3] Re-ranking with artifact regularization...")
    
    match_full = build_match_table(artifacts, weights)
    board, score = rerank_with_artifacts(
        artifacts, match_rgb, match_full, board, grid_size=4, verbose=verbose
    )
    
    # Phase 4: DISABLED - too slow, minimal benefit
    # board, score = multi_strategy_refinement(...)
    
    # Report final scores
    rgb_score = score_board(match_rgb, board, grid_size=4)
    
    if verbose:
        print(f"\n  Final RGB score: {rgb_score:.4f}")
    
    arrangement = board_to_arrangement(board, grid_size=4)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Final arrangement: {arrangement}")
        print(f"Final RGB score: {rgb_score:.4f}")
        print("=" * 60)
    
    return board, arrangement, rgb_score  # Return RGB score for comparison with legacy


def board_to_arrangement(board: dict, grid_size: int = 4) -> list:
    """Convert board dict to flat arrangement list."""
    return [board[(r, c)] for r in range(grid_size) for c in range(grid_size)]


def arrangement_to_board(arrangement: list, grid_size: int = 4) -> dict:
    """Convert flat arrangement to board dict."""
    board = {}
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            board[(r, c)] = arrangement[idx]
            idx += 1
    return board
