"""
Phase 2 Utility Functions for Puzzle Assembly
Includes MST-based global optimization with proper border reversal

This module contains core functions for:
- Edge profile extraction with proper reversal
- Fuzzy edge similarity comparison
- Piece compatibility analysis with hybrid scoring
- MST-based puzzle assembly with local refinement
"""

import numpy as np
import cv2
from collections import defaultdict


# ========== Edge Profile Extraction ==========

def extract_edge_profiles(edge_image):
    """
    Extracts 1D edge profiles from all 4 sides of a piece.
    
    Args:
        edge_image: 2D numpy array (binary edge-detected image)
    
    Returns:
        dict with keys: 'top', 'right', 'bottom', 'left'
        Each value is a 1D numpy array representing edge intensity along that side
    """
    profiles = {
        'top': edge_image[0, :].astype(np.float32),       # Top row
        'bottom': edge_image[-1, :].astype(np.float32),   # Bottom row
        'left': edge_image[:, 0].astype(np.float32),      # Left column
        'right': edge_image[:, -1].astype(np.float32)     # Right column
    }
    return profiles


def reverse_profile(profile):
    """
    Reverse a profile for proper border matching.
    CRITICAL: Left edge must match Right edge REVERSED (and vice versa).
    """
    return profile[::-1].copy()


def smooth_profile(profile, sigma=3):
    """
    Applies 1D Gaussian smoothing to a binary profile.
    This makes matching robust to small misalignments (1-3 pixels).
    
    Args:
        profile: 1D numpy array (binary 0/255 or 0/1)
        sigma: Spread of the blur (higher = more tolerant of misalignment)
    
    Returns:
        Smoothed profile as 1D numpy array
    """
    # Create a 1D Gaussian kernel
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    kernel = cv2.getGaussianKernel(ksize, sigma)
    
    # Reshape for 1D convolution
    return cv2.filter2D(profile.reshape(1, -1), -1, kernel).flatten()


# ========== Enhanced Edge Feature Extraction ==========

def extract_edge_strip_features(edge_image, strip_width=10):
    """
    Extract enhanced edge features from border strips (not just 1-pixel edges).
    Includes edge density, contour info, and structure.
    
    Args:
        edge_image: 2D binary edge image
        strip_width: Width of strip to extract from each border
    
    Returns:
        dict with edge features for each side
    """
    h, w = edge_image.shape
    
    features = {}
    
    # Extract strips
    strips = {
        'top': edge_image[:strip_width, :],
        'bottom': edge_image[-strip_width:, :],
        'left': edge_image[:, :strip_width],
        'right': edge_image[:, -strip_width:]
    }
    
    for side, strip in strips.items():
        # Edge density profile
        if side in ['top', 'bottom']:
            # Horizontal strip - sum along height for each column
            density_profile = np.sum(strip > 0, axis=0).astype(np.float32)
        else:
            # Vertical strip - sum along width for each row
            density_profile = np.sum(strip > 0, axis=1).astype(np.float32)
        
        # Normalize
        density_profile = density_profile / (strip_width + 1e-6)
        
        features[side] = {
            'strip': strip,
            'density_profile': density_profile,
            'total_density': np.mean(density_profile)
        }
    
    return features


def get_border_pixels(image, direction, border_width=3):
    """
    Extract border pixels from an image for pixel-level matching.
    
    Args:
        image: 2D or 3D numpy array (grayscale or RGB)
        direction: 'top', 'bottom', 'left', or 'right'
        border_width: Number of pixels to extract from border
    
    Returns:
        Border region as numpy array
    """
    if direction == 'top':
        return image[:border_width, :].copy()
    elif direction == 'bottom':
        return image[-border_width:, :].copy()
    elif direction == 'left':
        return image[:, :border_width].copy()
    elif direction == 'right':
        return image[:, -border_width:].copy()


def calculate_hybrid_score(edge_i, edge_j, color_i, color_j, direction, complements):
    """
    Calculate hybrid score combining edge matching and pixel matching.
    CRITICAL FIX: Properly reverses complementary borders for matching.
    
    Args:
        edge_i, edge_j: Edge images for pieces i and j
        color_i, color_j: Original color/grayscale images for pieces i and j
        direction: Direction from piece i to piece j
        complements: Dictionary of complementary directions
    
    Returns:
        Combined score (higher = better match)
    """
    complement = complements[direction]
    
    # 1. Edge Score (30% weight)
    feats_i = extract_edge_strip_features(edge_i, strip_width=10)
    feats_j = extract_edge_strip_features(edge_j, strip_width=10)
    
    # CRITICAL FIX: Reverse the density profile for proper matching
    profile_i = feats_i[direction]['density_profile']
    profile_j = reverse_profile(feats_j[complement]['density_profile'])
    
    # Correlation on reversed profiles
    min_len = min(len(profile_i), len(profile_j))
    p1 = profile_i[:min_len]
    p2 = profile_j[:min_len]
    
    if np.std(p1) > 1e-6 and np.std(p2) > 1e-6:
        edge_score = np.corrcoef(p1, p2)[0, 1]
        if np.isnan(edge_score):
            edge_score = 0.0
    else:
        edge_score = 0.0
    
    # 2. Pixel Score (70% weight) - borders must also be reversed
    border_i = get_border_pixels(color_i, direction, border_width=3)
    border_j_raw = get_border_pixels(color_j, complement, border_width=3)
    
    # CRITICAL FIX: Reverse the border pixels appropriately
    if direction in ['left', 'right']:
        # Vertical borders - reverse along height
        border_j = border_j_raw[::-1, :].copy()
    else:  # top, bottom
        # Horizontal borders - reverse along width
        border_j = border_j_raw[:, ::-1].copy()
    
    # Flatten for comparison
    if len(border_i.shape) == 3:
        # RGB - compare each channel
        border_i_flat = border_i.reshape(-1, border_i.shape[2]).astype(np.float32)
        border_j_flat = border_j.reshape(-1, border_j.shape[2]).astype(np.float32)
    else:
        # Grayscale
        border_i_flat = border_i.flatten().astype(np.float32)
        border_j_flat = border_j.flatten().astype(np.float32)
    
    # Ensure same length
    min_len = min(len(border_i_flat), len(border_j_flat))
    border_i_flat = border_i_flat[:min_len]
    border_j_flat = border_j_flat[:min_len]
    
    # Normalized Cross-Correlation
    if np.std(border_i_flat) > 1e-5 and np.std(border_j_flat) > 1e-5:
        if len(border_i_flat.shape) == 2:
            # For RGB, average correlation across channels
            correlations = []
            for ch in range(border_i_flat.shape[1]):
                bi_norm = (border_i_flat[:, ch] - np.mean(border_i_flat[:, ch])) / (np.std(border_i_flat[:, ch]) + 1e-5)
                bj_norm = (border_j_flat[:, ch] - np.mean(border_j_flat[:, ch])) / (np.std(border_j_flat[:, ch]) + 1e-5)
                correlations.append(np.mean(bi_norm * bj_norm))
            pixel_score = np.mean(correlations)
        else:
            # For grayscale
            bi_norm = (border_i_flat - np.mean(border_i_flat)) / (np.std(border_i_flat) + 1e-5)
            bj_norm = (border_j_flat - np.mean(border_j_flat)) / (np.std(border_j_flat) + 1e-5)
            pixel_score = np.mean(bi_norm * bj_norm)
    else:
        # If no variation, use SSD instead
        pixel_score = -np.mean((border_i_flat - border_j_flat) ** 2) / 10000.0
    
    # 3. Weighted Combination
    final_score = 0.3 * edge_score + 0.7 * pixel_score
    
    return final_score


def build_hybrid_compatibility_matrix(edge_images, color_images):
    """
    Build compatibility matrix combining edge features and pixel matching.
    This is the most robust approach with proper border reversal.
    
    Args:
        edge_images: List of edge-detected images
        color_images: List of original color/grayscale images
    
    Returns:
        compatibility matrix with hybrid scores (higher = better match)
    """
    num_pieces = len(edge_images)
    
    # Build compatibility matrix
    compatibility = defaultdict(lambda: defaultdict(dict))
    
    # Complementary sides for matching
    complements = {
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }
    
    # Calculate all pairwise compatibilities
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:
                continue
            
            for direction in ['top', 'right', 'bottom', 'left']:
                score = calculate_hybrid_score(
                    edge_images[i], edge_images[j],
                    color_images[i], color_images[j],
                    direction, complements
                )
                compatibility[i][j][direction] = score
    
    return compatibility


# ========== MST-Based Assembly ==========

def build_match_graph(compatibility, num_pieces, top_k=3):
    """
    Build a graph of top-K best matches for each piece-side pair.
    Returns weighted edges for global optimization.
    
    Args:
        compatibility: Compatibility matrix
        num_pieces: Number of pieces
        top_k: Number of top matches to keep per side
    
    Returns:
        List of (piece_i, piece_j, direction, score) tuples
    """
    edges = []
    
    for piece_i in range(num_pieces):
        for direction in ['top', 'right', 'bottom', 'left']:
            # Get all scores for this piece-side
            candidates = []
            for piece_j in range(num_pieces):
                if piece_i == piece_j:
                    continue
                score = compatibility[piece_i][piece_j][direction]
                candidates.append((piece_j, score))
            
            # Sort and take top K
            candidates.sort(key=lambda x: x[1], reverse=True)
            for piece_j, score in candidates[:top_k]:
                edges.append((piece_i, piece_j, direction, score))
    
    return edges


def assemble_puzzle_mst(compatibility, grid_size):
    """
    MST-based global assembly using Minimum Spanning Tree.
    MUCH more robust than greedy/backtracking for large puzzles.
    
    Args:
        compatibility: Compatibility matrix
        grid_size: Grid size
    
    Returns:
        2D numpy array with piece indices
    """
    num_pieces = grid_size * grid_size
    
    # 1. Build match graph with top-3 matches per side
    edges = build_match_graph(compatibility, num_pieces, top_k=3)
    
    # 2. Build MST using Kruskal's algorithm (union-find)
    # Use (1 - score) as cost so higher scores = lower cost
    edges_with_cost = [(1.0 - score, i, j, direction) for i, j, direction, score in edges]
    edges_with_cost.sort()  # Sort by cost
    
    # Union-Find
    parent = list(range(num_pieces))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Build MST edges
    mst_edges = []
    for cost, i, j, direction in edges_with_cost:
        if union(i, j):
            mst_edges.append((i, j, direction, 1.0 - cost))
        if len(mst_edges) == num_pieces - 1:
            break
    
    # 3. Build adjacency list from MST
    adjacency = defaultdict(list)
    for i, j, direction, score in mst_edges:
        adjacency[i].append((j, direction))
        # Add reverse edge
        complements = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        adjacency[j].append((i, complements[direction]))
    
    # 4. Find root (highest degree node)
    root = max(range(num_pieces), key=lambda x: len(adjacency[x]))
    
    # 5. BFS placement from root
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    placed = {}
    
    # Place root at center
    center_r, center_c = grid_size // 2, grid_size // 2
    grid[center_r, center_c] = root
    placed[root] = (center_r, center_c)
    
    queue = [root]
    
    while queue:
        current = queue.pop(0)
        curr_r, curr_c = placed[current]
        
        for neighbor, direction in adjacency[current]:
            if neighbor in placed:
                continue
            
            # Calculate neighbor position
            if direction == 'right' and curr_c + 1 < grid_size:
                new_r, new_c = curr_r, curr_c + 1
            elif direction == 'left' and curr_c > 0:
                new_r, new_c = curr_r, curr_c - 1
            elif direction == 'bottom' and curr_r + 1 < grid_size:
                new_r, new_c = curr_r + 1, curr_c
            elif direction == 'top' and curr_r > 0:
                new_r, new_c = curr_r - 1, curr_c
            else:
                continue
            
            if grid[new_r, new_c] == -1:
                grid[new_r, new_c] = neighbor
                placed[neighbor] = (new_r, new_c)
                queue.append(neighbor)
    
    # 6. Local refinement - swap pieces to improve total score
    improved = True
    iterations = 0
    max_iterations = 20
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for r1 in range(grid_size):
            for c1 in range(grid_size):
                piece1 = grid[r1, c1]
                if piece1 == -1:
                    continue
                
                # Try swapping with other pieces
                for r2 in range(grid_size):
                    for c2 in range(grid_size):
                        if (r1, c1) >= (r2, c2):
                            continue
                        
                        piece2 = grid[r2, c2]
                        if piece2 == -1:
                            continue
                        
                        # Calculate current scores
                        current_score = 0
                        swap_score = 0
                        
                        # Score piece1 at (r1, c1) and piece2 at (r2, c2)
                        for r, c, piece in [(r1, c1, piece1), (r2, c2, piece2)]:
                            if r > 0 and grid[r-1, c] != -1:
                                current_score += compatibility[piece][grid[r-1, c]]['top']
                            if r < grid_size-1 and grid[r+1, c] != -1:
                                current_score += compatibility[piece][grid[r+1, c]]['bottom']
                            if c > 0 and grid[r, c-1] != -1:
                                current_score += compatibility[piece][grid[r, c-1]]['left']
                            if c < grid_size-1 and grid[r, c+1] != -1:
                                current_score += compatibility[piece][grid[r, c+1]]['right']
                        
                        # Score after swap
                        for r, c, piece in [(r1, c1, piece2), (r2, c2, piece1)]:
                            if r > 0 and grid[r-1, c] != -1 and grid[r-1, c] not in [piece1, piece2]:
                                swap_score += compatibility[piece][grid[r-1, c]]['top']
                            if r < grid_size-1 and grid[r+1, c] != -1 and grid[r+1, c] not in [piece1, piece2]:
                                swap_score += compatibility[piece][grid[r+1, c]]['bottom']
                            if c > 0 and grid[r, c-1] != -1 and grid[r, c-1] not in [piece1, piece2]:
                                swap_score += compatibility[piece][grid[r, c-1]]['left']
                            if c < grid_size-1 and grid[r, c+1] != -1 and grid[r, c+1] not in [piece1, piece2]:
                                swap_score += compatibility[piece][grid[r, c+1]]['right']
                        
                        # Swap if improvement
                        if swap_score > current_score:
                            grid[r1, c1] = piece2
                            grid[r2, c2] = piece1
                            improved = True
    
    return grid


# ========== Visualization Helpers ==========

def create_assembled_image(grid, patches):
    """
    Reconstruct full image from assembled grid and piece patches.
    
    Args:
        grid: 2D array of piece indices
        patches: List of image patches (RGB or grayscale numpy arrays)
    
    Returns:
        Assembled image as numpy array
    """
    grid_size = grid.shape[0]
    
    # Determine if patches are color or grayscale
    if len(patches[0].shape) == 3:
        patch_h, patch_w, channels = patches[0].shape
        assembled = np.zeros((grid_size * patch_h, grid_size * patch_w, channels), 
                           dtype=patches[0].dtype)
    else:
        patch_h, patch_w = patches[0].shape
        assembled = np.zeros((grid_size * patch_h, grid_size * patch_w), 
                           dtype=patches[0].dtype)
    
    for row in range(grid_size):
        for col in range(grid_size):
            piece_idx = grid[row, col]
            if piece_idx >= 0 and piece_idx < len(patches):
                y_start = row * patch_h
                y_end = (row + 1) * patch_h
                x_start = col * patch_w
                x_end = (col + 1) * patch_w
                assembled[y_start:y_end, x_start:x_end] = patches[piece_idx]
    
    return assembled
