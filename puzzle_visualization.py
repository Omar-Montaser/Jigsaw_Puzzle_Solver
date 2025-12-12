"""
Visualization Module for Puzzle Assembly - Phase 2

Functions for visualizing edge matches, puzzle assembly, and debugging.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import json
from pathlib import Path


def visualize_edge_matches(matches, all_pieces_images, grid_size, 
                          max_matches=10, figsize=(20, 12)):
    """
    Visualize matching edges by drawing connections between pieces.
    
    Args:
        matches: list of (edge1, edge2, score) tuples
        all_pieces_images: dictionary mapping piece_id to original piece image
        grid_size: original grid size (2, 4, or 8)
        max_matches: maximum number of matches to display
        figsize: figure size
    """
    if not matches:
        print("No matches to visualize")
        return
    
    # Show top matches
    top_matches = matches[:max_matches]
    
    n_rows = min(len(top_matches), max_matches)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (edge1, edge2, score) in enumerate(top_matches):
        if idx >= n_rows:
            break
        
        # Get piece images
        img1 = all_pieces_images.get(edge1.piece_id)
        img2 = all_pieces_images.get(edge2.piece_id)
        
        if img1 is None or img2 is None:
            continue
        
        # Draw first piece with highlighted edge
        ax1 = axes[idx, 0]
        img1_vis = img1.copy()
        
        # Highlight the matching edge
        edge_color = (255, 0, 0)  # Red
        if len(edge1.contour_points) > 1:
            pts = edge1.contour_points.astype(np.int32)
            cv2.polylines(img1_vis, [pts], False, edge_color, 3)
        
        ax1.imshow(img1_vis)
        ax1.set_title(f"{edge1.piece_id}\n{edge1.edge_position} edge", fontsize=10)
        ax1.axis('off')
        
        # Draw second piece with highlighted edge
        ax2 = axes[idx, 1]
        img2_vis = img2.copy()
        
        if len(edge2.contour_points) > 1:
            pts = edge2.contour_points.astype(np.int32)
            cv2.polylines(img2_vis, [pts], False, edge_color, 3)
        
        ax2.imshow(img2_vis)
        ax2.set_title(f"{edge2.piece_id}\n{edge2.edge_position} edge", fontsize=10)
        ax2.axis('off')
        
        # Add score annotation
        fig.text(0.5, 0.98 - (idx * 1.0 / n_rows), 
                f'Match Score: {score:.3f}', 
                ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.suptitle('Top Edge Matches', fontsize=14, y=0.995)
    plt.show()


def visualize_puzzle_graph(matches, grid_size, all_pieces_images, 
                          confidence_threshold=5.0, figsize=(16, 16)):
    """
    Visualize puzzle pieces in a graph layout with connections showing matches.
    
    Args:
        matches: list of (edge1, edge2, score) tuples
        grid_size: size of original grid
        all_pieces_images: dict mapping piece_id to image
        confidence_threshold: only show matches below this threshold
        figsize: figure size
    """
    num_pieces = grid_size * grid_size
    
    # Filter matches by confidence
    confident_matches = [(e1, e2, s) for e1, e2, s in matches 
                        if s < confidence_threshold]
    
    if not confident_matches:
        print(f"No confident matches found (threshold: {confidence_threshold})")
        return
    
    # Create grid layout for pieces
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position pieces in a grid
    piece_positions = {}
    piece_size = 1.5
    spacing = 2.0
    
    for i in range(grid_size):
        for j in range(grid_size):
            piece_num = i * grid_size + j
            x = j * spacing
            y = (grid_size - 1 - i) * spacing
            piece_id = None
            
            # Find matching piece_id
            for pid in all_pieces_images.keys():
                if f"piece_{piece_num:02d}" in pid:
                    piece_id = pid
                    break
            
            if piece_id:
                piece_positions[piece_id] = (x, y)
                
                # Draw piece thumbnail
                img = all_pieces_images[piece_id]
                if img is not None:
                    extent = [x - piece_size/2, x + piece_size/2, 
                            y - piece_size/2, y + piece_size/2]
                    ax.imshow(img, extent=extent, zorder=1)
                
                # Add piece label
                ax.text(x, y - piece_size/2 - 0.2, 
                       f"P{piece_num}", 
                       ha='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw match connections
    for edge1, edge2, score in confident_matches:
        pos1 = piece_positions.get(edge1.piece_id)
        pos2 = piece_positions.get(edge2.piece_id)
        
        if pos1 and pos2:
            # Color based on confidence (green = good, red = poor)
            confidence = 1.0 - min(score / confidence_threshold, 1.0)
            color = plt.cm.RdYlGn(confidence)
            
            # Draw arrow
            arrow = FancyArrowPatch(
                pos1, pos2,
                arrowstyle='-',
                linewidth=2 + confidence * 3,
                color=color,
                alpha=0.6,
                zorder=0
            )
            ax.add_patch(arrow)
            
            # Add score label at midpoint
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax.text(mid_x, mid_y, f'{score:.2f}', 
                   fontsize=7, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-1, grid_size * spacing)
    ax.set_ylim(-1, grid_size * spacing)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Puzzle Match Graph ({len(confident_matches)} confident matches)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()


def visualize_edge_descriptors(edge, figsize=(15, 4)):
    """
    Visualize the shape descriptors of a single edge for debugging.
    
    Args:
        edge: PuzzleEdge object
        figsize: figure size
    """
    if edge.descriptors is None:
        print("No descriptors available")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Plot contour
    axes[0].plot(edge.contour_points[:, 0], edge.contour_points[:, 1], 'b-', linewidth=2)
    axes[0].scatter(edge.contour_points[:, 0], edge.contour_points[:, 1], 
                   c='red', s=10, alpha=0.5)
    axes[0].set_title(f'{edge.piece_id}\n{edge.edge_position} edge')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Fourier descriptors
    if 'fourier' in edge.descriptors:
        fourier = edge.descriptors['fourier']
        axes[1].bar(range(len(fourier)), fourier)
        axes[1].set_title('Fourier Descriptors')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True, alpha=0.3)
    
    # Plot curvature signature
    if 'curvature' in edge.descriptors:
        curvature = edge.descriptors['curvature']
        axes[2].plot(curvature, linewidth=2)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].set_title('Curvature Signature')
        axes[2].set_xlabel('Position along edge')
        axes[2].set_ylabel('Curvature')
        axes[2].grid(True, alpha=0.3)
    
    # Plot centroid distance signature
    if 'centroid_distances' in edge.descriptors:
        distances = edge.descriptors['centroid_distances']
        axes[3].plot(distances, linewidth=2)
        axes[3].set_title('Centroid Distance Signature')
        axes[3].set_xlabel('Position along edge')
        axes[3].set_ylabel('Normalized Distance')
        axes[3].grid(True, alpha=0.3)
    
    # Add edge properties
    properties_text = f"Border: {edge.is_border}\n"
    if 'straightness' in edge.descriptors:
        properties_text += f"Straightness: {edge.descriptors['straightness']:.3f}"
    
    fig.text(0.02, 0.02, properties_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def create_assembly_visualization(piece_positions, all_pieces_images, 
                                 grid_size, piece_size=200):
    """
    Create a visualization of the assembled puzzle.
    
    Args:
        piece_positions: dict mapping piece_id to (row, col) position
        all_pieces_images: dict mapping piece_id to image
        grid_size: size of the puzzle grid
        piece_size: size of each piece in pixels
        
    Returns:
        assembled image as numpy array
    """
    # Create canvas
    canvas_size = grid_size * piece_size
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    
    # Place pieces
    for piece_id, (row, col) in piece_positions.items():
        img = all_pieces_images.get(piece_id)
        
        if img is not None:
            # Resize piece to standard size
            resized = cv2.resize(img, (piece_size, piece_size))
            
            # Calculate position
            y_start = row * piece_size
            y_end = y_start + piece_size
            x_start = col * piece_size
            x_end = x_start + piece_size
            
            # Place on canvas
            canvas[y_start:y_end, x_start:x_end] = resized
            
            # Draw border
            cv2.rectangle(canvas, (x_start, y_start), (x_end-1, y_end-1), 
                        (100, 100, 100), 2)
            
            # Add piece label
            label = piece_id.split('_')[-1]
            cv2.putText(canvas, f"P{label}", 
                       (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return canvas


def visualize_assembly_result(piece_positions, all_pieces_images, 
                              grid_size, matches=None, figsize=(12, 12)):
    """
    Visualize the final or partial puzzle assembly.
    
    Args:
        piece_positions: dict mapping piece_id to (row, col) position
        all_pieces_images: dict mapping piece_id to image
        grid_size: size of the puzzle grid
        matches: optional list of matches to overlay
        figsize: figure size
    """
    assembled = create_assembly_visualization(piece_positions, all_pieces_images, grid_size)
    
    plt.figure(figsize=figsize)
    plt.imshow(assembled)
    plt.title(f'Puzzle Assembly ({len(piece_positions)}/{grid_size*grid_size} pieces placed)', 
             fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_match_report(matches, output_path, top_n=50):
    """
    Save a detailed report of edge matches to a JSON file.
    
    Args:
        matches: list of (edge1, edge2, score) tuples
        output_path: path to save the report
        top_n: number of top matches to include
    """
    report = {
        'total_matches': len(matches),
        'top_matches': []
    }
    
    for i, (edge1, edge2, score) in enumerate(matches[:top_n]):
        match_info = {
            'rank': i + 1,
            'score': float(score),
            'piece1': {
                'id': edge1.piece_id,
                'edge': edge1.edge_position,
                'is_border': edge1.is_border
            },
            'piece2': {
                'id': edge2.piece_id,
                'edge': edge2.edge_position,
                'is_border': edge2.is_border
            }
        }
        report['top_matches'].append(match_info)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Match report saved to {output_path}")


def load_piece_images(processed_dir, puzzle_type, image_id):
    """
    Load original piece images for visualization.
    
    Args:
        processed_dir: base directory of processed artifacts
        puzzle_type: 'puzzle_2x2', 'puzzle_4x4', or 'puzzle_8x8'
        image_id: image identifier
        
    Returns:
        Dictionary mapping piece_id to BGR image
    """
    pieces = {}
    enhanced_dir = Path(processed_dir) / puzzle_type / f"image_{image_id}" / "enhanced_pieces"
    
    if not enhanced_dir.exists():
        print(f"Directory not found: {enhanced_dir}")
        return pieces
    
    # Load all enhanced piece images
    piece_files = sorted(enhanced_dir.glob("piece_*_enhanced.png"))
    
    for piece_file in piece_files:
        # Extract piece number
        piece_num = piece_file.stem.split('_')[1]
        piece_id = f"image_{image_id}_piece_{piece_num}"
        
        # Load image (grayscale)
        img = cv2.imread(str(piece_file), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Convert to BGR for consistent display
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            pieces[piece_id] = img_bgr
    
    return pieces
