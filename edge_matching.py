"""
Edge Matching and Puzzle Assembly Module for Phase 2

This module provides functions for:
1. Extracting edges from puzzle pieces
2. Representing edges using shape descriptors
3. Comparing and matching complementary edges
4. Assembling puzzle pieces based on matches
"""

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import distance_transform_edt
import json
from pathlib import Path


class PuzzleEdge:
    """Represents a single edge of a puzzle piece."""
    
    def __init__(self, piece_id, edge_position, contour_points, is_border=False, texture_profile=None):
        """
        Args:
            piece_id: identifier for the piece (e.g., "image_0_piece_0")
            edge_position: 'top', 'bottom', 'left', or 'right'
            contour_points: numpy array of (x, y) points along the edge
            is_border: True if this is a flat border edge
            texture_profile: intensity profile along the edge for texture matching
        """
        self.piece_id = piece_id
        self.edge_position = edge_position
        self.contour_points = contour_points
        self.is_border = is_border
        self.texture_profile = texture_profile
        
        # Compute shape descriptors
        self.descriptors = self._compute_descriptors()
    
    def _compute_descriptors(self):
        """Compute rotation-invariant shape descriptors for the edge."""
        if len(self.contour_points) < 5:
            return None
        
        descriptors = {}
        
        # 1. Fourier Descriptors (rotation-invariant)
        descriptors['fourier'] = self._compute_fourier_descriptors()
        
        # 2. Curvature signature
        descriptors['curvature'] = self._compute_curvature_signature()
        
        # 3. Distance from centroid signature
        descriptors['centroid_distances'] = self._compute_centroid_distance_signature()
        
        # 4. Edge straightness (for border detection)
        descriptors['straightness'] = self._compute_straightness()
        
        return descriptors
    
    def _compute_fourier_descriptors(self, num_descriptors=20):
        """Compute Fourier descriptors for shape representation."""
        # Convert to complex representation
        points = self.contour_points
        if len(points) < 2:
            return np.zeros(num_descriptors)
        
        # Create complex coordinates
        complex_points = points[:, 0] + 1j * points[:, 1]
        
        # Compute FFT
        fft_result = np.fft.fft(complex_points)
        
        # Take magnitude to achieve rotation invariance
        # Normalize by the DC component (exclude index 0)
        if len(fft_result) > 1 and abs(fft_result[1]) > 1e-10:
            normalized = np.abs(fft_result[1:num_descriptors+1]) / abs(fft_result[1])
            
            # Pad if needed
            if len(normalized) < num_descriptors:
                normalized = np.pad(normalized, (0, num_descriptors - len(normalized)))
            
            return normalized[:num_descriptors]
        
        return np.zeros(num_descriptors)
    
    def _compute_curvature_signature(self, num_points=50):
        """Compute curvature at sampled points along the edge."""
        points = self.contour_points
        
        if len(points) < 10:
            return np.zeros(num_points)
        
        # Resample to fixed number of points
        indices = np.linspace(0, len(points)-1, num_points, dtype=int)
        sampled = points[indices]
        
        # Compute curvature using finite differences
        curvatures = []
        for i in range(1, len(sampled)-1):
            p1, p2, p3 = sampled[i-1], sampled[i], sampled[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product gives curvature direction
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            
            # Normalize by distance
            dist = np.linalg.norm(v1) * np.linalg.norm(v2)
            if dist > 1e-10:
                curvature = cross / dist
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        # Pad to match num_points
        curvatures = [0] + curvatures + [0]
        if len(curvatures) < num_points:
            curvatures.extend([0] * (num_points - len(curvatures)))
        
        return np.array(curvatures[:num_points])
    
    def _compute_centroid_distance_signature(self, num_points=50):
        """Compute distances from edge points to centroid."""
        points = self.contour_points
        
        if len(points) < 2:
            return np.zeros(num_points)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Resample to fixed number of points
        indices = np.linspace(0, len(points)-1, num_points, dtype=int)
        sampled = points[indices]
        
        # Compute distances
        distances = np.linalg.norm(sampled - centroid, axis=1)
        
        # Normalize
        if np.max(distances) > 1e-10:
            distances = distances / np.max(distances)
        
        return distances
    
    def _compute_straightness(self):
        """Compute how straight the edge is (for border detection)."""
        if len(self.contour_points) < 2:
            return 0
        
        # Fit a line
        points = self.contour_points
        if len(points) < 2:
            return 0
        
        # Calculate deviation from line
        start = points[0]
        end = points[-1]
        
        # Distance of each point from line
        total_deviation = 0
        line_length = np.linalg.norm(end - start)
        
        if line_length < 1e-10:
            return 1.0
        
        for point in points:
            # Distance from point to line
            deviation = np.abs(np.cross(end - start, start - point)) / line_length
            total_deviation += deviation
        
        # Normalize by number of points and line length
        avg_deviation = total_deviation / (len(points) * line_length)
        
        # Convert to straightness (0 = curved, 1 = straight)
        straightness = 1.0 / (1.0 + avg_deviation * 10)
        
        return straightness


def extract_edge_texture(piece_image, edge_points, edge_position, strip_width=10):
    """
    Extract texture profile along an edge for content-based matching.
    
    Args:
        piece_image: grayscale image of the piece
        edge_points: array of (x, y) points along the edge
        edge_position: 'top', 'bottom', 'left', or 'right'
        strip_width: width of the strip to sample along the edge
        
    Returns:
        Array of intensity values along the edge
    """
    if piece_image is None or len(edge_points) == 0:
        return None
    
    h, w = piece_image.shape[:2]
    texture_values = []
    
    # Sample intensity values along the edge
    for point in edge_points:
        x, y = int(point[0]), int(point[1])
        
        # Sample a strip perpendicular to the edge
        if edge_position in ['top', 'bottom']:
            # Sample vertically
            if edge_position == 'top':
                y_start = max(0, y)
                y_end = min(h, y + strip_width)
            else:
                y_start = max(0, y - strip_width)
                y_end = min(h, y)
            
            if 0 <= x < w and y_start < y_end:
                strip = piece_image[y_start:y_end, max(0, x):min(w, x+1)]
                if strip.size > 0:
                    texture_values.append(np.mean(strip))
        else:
            # Sample horizontally
            if edge_position == 'right':
                x_start = max(0, x - strip_width)
                x_end = min(w, x)
            else:
                x_start = max(0, x)
                x_end = min(w, x + strip_width)
            
            if 0 <= y < h and x_start < x_end:
                strip = piece_image[max(0, y):min(h, y+1), x_start:x_end]
                if strip.size > 0:
                    texture_values.append(np.mean(strip))
    
    if not texture_values:
        return None
    
    # Normalize to 0-1 range
    texture_array = np.array(texture_values)
    if np.max(texture_array) > 0:
        texture_array = texture_array / 255.0
    
    return texture_array


def extract_piece_edges(edge_image, piece_id, min_contour_length=20, original_piece_image=None):
    """
    Extract the four edges (top, bottom, left, right) from a puzzle piece edge image.
    
    Args:
        edge_image: binary edge image from Phase 1
        piece_id: identifier for this piece
        min_contour_length: minimum number of points in a valid edge contour
        original_piece_image: original grayscale piece image for texture extraction
        
    Returns:
        Dictionary mapping edge positions to PuzzleEdge objects
    """
    # Find contours
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return {}
    
    # Get the largest contour (main piece boundary)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Simplify slightly to reduce noise
    epsilon = 0.001 * cv2.arcLength(main_contour, True)
    main_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Reshape contour
    points = main_contour.reshape(-1, 2)
    
    if len(points) < 4:
        return {}
    
    # Get bounding box to determine edge regions
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Define edge regions with some tolerance
    tolerance = max(w, h) * 0.15
    
    edges = {}
    
    # Classify points into edges based on position
    top_points = points[points[:, 1] < y + tolerance]
    bottom_points = points[points[:, 1] > y + h - tolerance]
    left_points = points[points[:, 0] < x + tolerance]
    right_points = points[points[:, 0] > x + w - tolerance]
    
    # Create PuzzleEdge objects
    if len(top_points) >= min_contour_length:
        # Sort by x-coordinate
        top_points = top_points[np.argsort(top_points[:, 0])]
        texture = extract_edge_texture(original_piece_image, top_points, 'top') if original_piece_image is not None else None
        edges['top'] = PuzzleEdge(piece_id, 'top', top_points, texture_profile=texture)
    
    if len(bottom_points) >= min_contour_length:
        # Sort by x-coordinate
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        texture = extract_edge_texture(original_piece_image, bottom_points, 'bottom') if original_piece_image is not None else None
        edges['bottom'] = PuzzleEdge(piece_id, 'bottom', bottom_points, texture_profile=texture)
    
    if len(left_points) >= min_contour_length:
        # Sort by y-coordinate
        left_points = left_points[np.argsort(left_points[:, 1])]
        texture = extract_edge_texture(original_piece_image, left_points, 'left') if original_piece_image is not None else None
        edges['left'] = PuzzleEdge(piece_id, 'left', left_points, texture_profile=texture)
    
    if len(right_points) >= min_contour_length:
        # Sort by y-coordinate
        right_points = right_points[np.argsort(right_points[:, 1])]
        texture = extract_edge_texture(original_piece_image, right_points, 'right') if original_piece_image is not None else None
        edges['right'] = PuzzleEdge(piece_id, 'right', right_points, texture_profile=texture)
    
    # Detect border edges (straight edges)
    for edge_pos, edge in edges.items():
        if edge.descriptors and edge.descriptors['straightness'] > 0.65:
            edge.is_border = True
    
    return edges


# ============================================================================
# NORMALIZATION AND RESAMPLING UTILITIES
# ============================================================================

def normalize_to_unit_length(arr):
    """Normalize array to unit length (L2 norm = 1). Returns zeros if input is zero."""
    arr = np.asarray(arr, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm > 1e-10:
        return arr / norm
    return np.zeros_like(arr)


def normalize_to_01(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val > 1e-10:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr)


def resample_signal(signal, target_len):
    """Resample a 1D signal to target length using linear interpolation."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        return np.zeros(target_len)
    if len(signal) == target_len:
        return signal
    
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal)


def resample_points(points, target_len):
    """Resample 2D points array to target length."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros((target_len, 2))
    
    x_resampled = resample_signal(points[:, 0], target_len)
    y_resampled = resample_signal(points[:, 1], target_len)
    return np.column_stack([x_resampled, y_resampled])


# ============================================================================
# DIAGNOSTICS GLOBALS
# ============================================================================

# Global flag to enable/disable diagnostics
ENABLE_DIAGNOSTICS = False
DIAGNOSTICS_OUTPUT_DIR = Path("./test_results/edge_diagnostics")


def set_diagnostics(enabled, output_dir=None):
    """Enable or disable diagnostics output."""
    global ENABLE_DIAGNOSTICS, DIAGNOSTICS_OUTPUT_DIR
    ENABLE_DIAGNOSTICS = enabled
    if output_dir:
        DIAGNOSTICS_OUTPUT_DIR = Path(output_dir)
    if enabled:
        DIAGNOSTICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_edge_comparison_debug_image(edge1, edge2, points1_resampled, points2_reversed, 
                                     score_details, output_path):
    """Save a debug image overlaying two compared edges after resampling and reversing."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original edges
    ax1 = axes[0]
    if len(edge1.contour_points) > 0:
        ax1.plot(edge1.contour_points[:, 0], edge1.contour_points[:, 1], 
                 'b-', linewidth=2, label=f'{edge1.piece_id} {edge1.edge_position}')
    if len(edge2.contour_points) > 0:
        ax1.plot(edge2.contour_points[:, 0], edge2.contour_points[:, 1], 
                 'r-', linewidth=2, label=f'{edge2.piece_id} {edge2.edge_position}')
    ax1.set_title('Original Edges')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # Plot 2: Resampled and aligned (edge2 reversed)
    ax2 = axes[1]
    if len(points1_resampled) > 0:
        ax2.plot(points1_resampled[:, 0], points1_resampled[:, 1], 
                 'b-', linewidth=2, label='Edge1 resampled')
        ax2.scatter(points1_resampled[0, 0], points1_resampled[0, 1], 
                    c='blue', s=100, marker='o', zorder=5)
    if len(points2_reversed) > 0:
        ax2.plot(points2_reversed[:, 0], points2_reversed[:, 1], 
                 'r--', linewidth=2, label='Edge2 reversed')
        ax2.scatter(points2_reversed[0, 0], points2_reversed[0, 1], 
                    c='red', s=100, marker='s', zorder=5)
    ax2.set_title('Resampled & Edge2 Reversed')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    # Plot 3: Score details
    ax3 = axes[2]
    ax3.axis('off')
    score_text = "Score Breakdown:\n" + "="*30 + "\n"
    for key, value in score_details.items():
        score_text += f"{key}: {value:.4f}\n"
    ax3.text(0.1, 0.9, score_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EDGE SIMILARITY FUNCTION (FIXED)
# ============================================================================

def compute_edge_similarity(edge1, edge2, weights=None, debug_label=None):
    """
    Compute similarity score between two edges.
    Lower score = better match (complementary edges).
    
    FIXES IMPLEMENTED:
    1. Correct scoring logic: lower = better, all distances non-negative
    2. Reverse one edge before comparison (edge2 is reversed)
    3. Normalize all descriptors to [0,1] or unit length
    4. Curvature complementarity: curvA + reversed(curvB)
    5. Clean weighted sum: weights sum to 1
    
    Args:
        edge1, edge2: PuzzleEdge objects
        weights: dict of weights for different features (must sum to 1)
        debug_label: optional string for debug output file naming
        
    Returns:
        similarity score (0 = perfect match, higher = less similar)
    """
    if edge1.descriptors is None or edge2.descriptors is None:
        return float('inf')
    
    # Fixed weights that sum to 1.0
    if weights is None:
        weights = {
            'fourier': 0.25,
            'curvature': 0.25,
            'centroid_distances': 0.25,
            'texture': 0.25
        }
    
    # Verify weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        # Normalize weights
        weights = {k: v / weight_sum for k, v in weights.items()}
    
    # Border edges can only match with border edges
    if edge1.is_border != edge2.is_border:
        return float('inf')
    
    # Border edges should have very low dissimilarity
    if edge1.is_border and edge2.is_border:
        return 0.0  # Perfect match for border edges
    
    # Store individual distances for diagnostics
    score_details = {}
    
    # Standard resampling length
    RESAMPLE_LEN = 50
    
    # ========================================================================
    # FOURIER DESCRIPTORS
    # ========================================================================
    fourier_dist = 0.0
    if 'fourier' in edge1.descriptors and 'fourier' in edge2.descriptors:
        fourier1 = np.asarray(edge1.descriptors['fourier'], dtype=np.float64)
        fourier2 = np.asarray(edge2.descriptors['fourier'], dtype=np.float64)
        
        # Normalize to unit length
        fourier1_norm = normalize_to_unit_length(fourier1)
        fourier2_norm = normalize_to_unit_length(fourier2)
        
        # L2 distance (max possible = 2 for unit vectors)
        fourier_dist = np.linalg.norm(fourier1_norm - fourier2_norm)
        # Normalize to [0, 1] range (max L2 distance between unit vectors is 2)
        fourier_dist = min(fourier_dist / 2.0, 1.0)
    
    score_details['fourier_dist'] = fourier_dist
    
    # ========================================================================
    # CURVATURE SIGNATURES (with proper reversal and complementarity)
    # ========================================================================
    curv_dist = 0.0
    if 'curvature' in edge1.descriptors and 'curvature' in edge2.descriptors:
        curv1 = np.asarray(edge1.descriptors['curvature'], dtype=np.float64)
        curv2 = np.asarray(edge2.descriptors['curvature'], dtype=np.float64)
        
        # Resample to same length
        curv1 = resample_signal(curv1, RESAMPLE_LEN)
        curv2 = resample_signal(curv2, RESAMPLE_LEN)
        
        # Normalize to unit length
        curv1_norm = normalize_to_unit_length(curv1)
        curv2_norm = normalize_to_unit_length(curv2)
        
        # CRITICAL FIX: Complementary edges have OPPOSITE curvature
        # Reverse edge2's curvature and check if curv1 + reversed(curv2) ≈ 0
        curv2_reversed = curv2_norm[::-1]
        
        # For perfect complementary edges: curv1 + curv2_reversed should be ~0
        curv_dist = np.linalg.norm(curv1_norm + curv2_reversed)
        # Normalize to [0, 1] (max is 2 for unit vectors)
        curv_dist = min(curv_dist / 2.0, 1.0)
    
    score_details['curvature_dist'] = curv_dist
    
    # ========================================================================
    # CENTROID DISTANCE SIGNATURES (with reversal)
    # ========================================================================
    centroid_dist = 0.0
    if 'centroid_distances' in edge1.descriptors and 'centroid_distances' in edge2.descriptors:
        cent1 = np.asarray(edge1.descriptors['centroid_distances'], dtype=np.float64)
        cent2 = np.asarray(edge2.descriptors['centroid_distances'], dtype=np.float64)
        
        # Resample to same length
        cent1 = resample_signal(cent1, RESAMPLE_LEN)
        cent2 = resample_signal(cent2, RESAMPLE_LEN)
        
        # Normalize to [0, 1] range
        cent1_norm = normalize_to_01(cent1)
        cent2_norm = normalize_to_01(cent2)
        
        # CRITICAL FIX: Reverse edge2's signature for proper endpoint alignment
        cent2_reversed = cent2_norm[::-1]
        
        # L2 distance
        centroid_dist = np.linalg.norm(cent1_norm - cent2_reversed)
        # Normalize (max L2 for [0,1] normalized vectors of length N is sqrt(N))
        centroid_dist = centroid_dist / np.sqrt(RESAMPLE_LEN)
        centroid_dist = min(centroid_dist, 1.0)
    
    score_details['centroid_dist'] = centroid_dist
    
    # ========================================================================
    # TEXTURE PROFILES (with reversal)
    # ========================================================================
    texture_dist = 0.0
    if edge1.texture_profile is not None and edge2.texture_profile is not None:
        tex1 = np.asarray(edge1.texture_profile, dtype=np.float64)
        tex2 = np.asarray(edge2.texture_profile, dtype=np.float64)
        
        if len(tex1) > 0 and len(tex2) > 0:
            # Resample to same length
            tex1 = resample_signal(tex1, RESAMPLE_LEN)
            tex2 = resample_signal(tex2, RESAMPLE_LEN)
            
            # Normalize to [0, 1] range
            tex1_norm = normalize_to_01(tex1)
            tex2_norm = normalize_to_01(tex2)
            
            # CRITICAL FIX: Reverse edge2's texture for proper alignment
            tex2_reversed = tex2_norm[::-1]
            
            # L2 distance
            texture_dist = np.linalg.norm(tex1_norm - tex2_reversed)
            # Normalize (max L2 for [0,1] normalized vectors of length N is sqrt(N))
            texture_dist = texture_dist / np.sqrt(RESAMPLE_LEN)
            texture_dist = min(texture_dist, 1.0)
    else:
        # If no texture, don't penalize but don't reward either
        texture_dist = 0.5  # Neutral score
    
    score_details['texture_dist'] = texture_dist
    
    # ========================================================================
    # WEIGHTED COMBINATION (weights sum to 1)
    # ========================================================================
    total_score = (
        weights['fourier'] * fourier_dist +
        weights['curvature'] * curv_dist +
        weights['centroid_distances'] * centroid_dist +
        weights['texture'] * texture_dist
    )
    
    score_details['total_score'] = total_score
    
    # ========================================================================
    # DIAGNOSTICS OUTPUT
    # ========================================================================
    if ENABLE_DIAGNOSTICS:
        # Print diagnostics
        print(f"\n{'='*60}")
        print(f"Edge Comparison: {edge1.piece_id}:{edge1.edge_position} vs {edge2.piece_id}:{edge2.edge_position}")
        print(f"  Fourier distance:    {fourier_dist:.4f} (weight: {weights['fourier']:.2f})")
        print(f"  Curvature distance:  {curv_dist:.4f} (weight: {weights['curvature']:.2f})")
        print(f"  Centroid distance:   {centroid_dist:.4f} (weight: {weights['centroid_distances']:.2f})")
        print(f"  Texture distance:    {texture_dist:.4f} (weight: {weights['texture']:.2f})")
        print(f"  TOTAL SCORE:         {total_score:.4f}")
        print(f"{'='*60}")
        
        # Save debug image
        if debug_label is None:
            debug_label = f"{edge1.piece_id}_{edge1.edge_position}_vs_{edge2.piece_id}_{edge2.edge_position}"
        
        # Prepare resampled points for visualization
        points1_resampled = resample_points(edge1.contour_points, RESAMPLE_LEN)
        points2_resampled = resample_points(edge2.contour_points, RESAMPLE_LEN)
        points2_reversed = points2_resampled[::-1]
        
        output_path = DIAGNOSTICS_OUTPUT_DIR / f"debug_{debug_label}.png"
        try:
            save_edge_comparison_debug_image(
                edge1, edge2, points1_resampled, points2_reversed,
                score_details, output_path
            )
        except Exception as e:
            print(f"Warning: Could not save debug image: {e}")
    
    return total_score


def find_edge_matches(all_edges, compatibility_threshold=10.0, top_k=3):
    """
    Find matching edges across all puzzle pieces.
    
    Args:
        all_edges: list of dictionaries, each containing edges for a piece
        compatibility_threshold: maximum distance for a valid match
        top_k: return top K matches for each edge
        
    Returns:
        List of match tuples: (edge1, edge2, similarity_score)
    """
    matches = []
    
    # Collect all non-border edges
    edge_list = []
    for piece_edges in all_edges:
        for edge_pos, edge in piece_edges.items():
            if not edge.is_border:
                edge_list.append((edge, piece_edges))
    
    print(f"Finding matches for {len(edge_list)} non-border edges...")
    
    # Compare each pair of edges
    for i, (edge1, piece1_edges) in enumerate(edge_list):
        candidates = []
        
        for j, (edge2, piece2_edges) in enumerate(edge_list):
            # Don't compare edge with itself or edges from same piece
            if i >= j or edge1.piece_id == edge2.piece_id:
                continue
            
            # Only compare compatible orientations
            # top matches with bottom, left matches with right
            if (edge1.edge_position == 'top' and edge2.edge_position == 'bottom') or \
               (edge1.edge_position == 'bottom' and edge2.edge_position == 'top') or \
               (edge1.edge_position == 'left' and edge2.edge_position == 'right') or \
               (edge1.edge_position == 'right' and edge2.edge_position == 'left'):
                
                similarity = compute_edge_similarity(edge1, edge2)
                
                if similarity < compatibility_threshold:
                    candidates.append((edge1, edge2, similarity))
        
        # Keep top K matches
        candidates.sort(key=lambda x: x[2])
        matches.extend(candidates[:top_k])
    
    # Remove duplicates and sort by similarity
    unique_matches = []
    seen = set()
    
    for match in sorted(matches, key=lambda x: x[2]):
        edge1, edge2, score = match
        pair_id = tuple(sorted([edge1.piece_id + edge1.edge_position, 
                                edge2.piece_id + edge2.edge_position]))
        
        if pair_id not in seen:
            seen.add(pair_id)
            unique_matches.append(match)
    
    print(f"Found {len(unique_matches)} unique potential matches")
    
    return unique_matches


def load_processed_pieces(processed_dir, puzzle_type, image_id):
    """
    Load processed edge images from Phase 1 output.
    
    Args:
        processed_dir: base directory of processed artifacts
        puzzle_type: 'puzzle_2x2', 'puzzle_4x4', or 'puzzle_8x8'
        image_id: image identifier
        
    Returns:
        List of (piece_id, edge_image, enhanced_image) tuples
    """
    pieces = []
    edge_dir = Path(processed_dir) / puzzle_type / f"image_{image_id}" / "edge_images"
    enhanced_dir = Path(processed_dir) / puzzle_type / f"image_{image_id}" / "enhanced_pieces"
    
    if not edge_dir.exists():
        print(f"Directory not found: {edge_dir}")
        return pieces
    
    # Load all edge images
    edge_files = sorted(edge_dir.glob("piece_*_edges.png"))
    
    for edge_file in edge_files:
        # Extract piece number
        piece_num = edge_file.stem.split('_')[1]
        piece_id = f"image_{image_id}_piece_{piece_num}"
        
        # Load edge image
        edge_image = cv2.imread(str(edge_file), cv2.IMREAD_GRAYSCALE)
        
        # Load enhanced image for texture
        enhanced_file = enhanced_dir / f"piece_{piece_num}_enhanced.png"
        enhanced_image = None
        if enhanced_file.exists():
            enhanced_image = cv2.imread(str(enhanced_file), cv2.IMREAD_GRAYSCALE)
        
        if edge_image is not None:
            pieces.append((piece_id, edge_image, enhanced_image))
    
    return pieces


def extract_all_edges_from_puzzle(processed_dir, puzzle_type, image_id):
    """
    Extract edges from all pieces of a puzzle image.
    
    Returns:
        List of dictionaries, each containing edges for one piece
    """
    pieces = load_processed_pieces(processed_dir, puzzle_type, image_id)
    
    all_edges = []
    
    print(f"Extracting edges from {len(pieces)} pieces...")
    
    for piece_id, edge_image, enhanced_image in pieces:
        edges = extract_piece_edges(edge_image, piece_id, original_piece_image=enhanced_image)
        
        if edges:
            all_edges.append(edges)
            print(f"  {piece_id}: found {len(edges)} edges", end="")
            border_count = sum(1 for e in edges.values() if e.is_border)
            if border_count > 0:
                print(f" ({border_count} border)", end="")
            print()
    
    return all_edges
