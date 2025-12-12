"""
Puzzle Assembly Module - Phase 2

Algorithms for assembling puzzle pieces based on edge matches.
"""

import numpy as np
from collections import defaultdict
import heapq


class PuzzleSolver:
    """
    Assembles puzzle pieces using edge matching results.
    Uses a greedy approach with backtracking.
    """
    
    def __init__(self, grid_size, matches):
        """
        Args:
            grid_size: size of puzzle grid (2, 4, or 8)
            matches: list of (edge1, edge2, score) tuples from edge matching
        """
        self.grid_size = grid_size
        self.matches = matches
        self.piece_positions = {}
        self.position_pieces = {}  # (row, col) -> piece_id
        
        # Build adjacency graph from matches
        self.build_match_graph()
    
    def build_match_graph(self):
        """Build a graph of piece connections based on matches."""
        self.match_graph = defaultdict(list)
        
        for edge1, edge2, score in self.matches:
            piece1 = edge1.piece_id
            piece2 = edge2.piece_id
            
            # Determine relative position
            relative_pos = self._get_relative_position(edge1.edge_position, 
                                                       edge2.edge_position)
            
            if relative_pos:
                self.match_graph[piece1].append({
                    'neighbor': piece2,
                    'direction': relative_pos,
                    'score': score,
                    'edge1': edge1,
                    'edge2': edge2
                })
                
                # Add reverse connection
                reverse_dir = self._reverse_direction(relative_pos)
                self.match_graph[piece2].append({
                    'neighbor': piece1,
                    'direction': reverse_dir,
                    'score': score,
                    'edge1': edge2,
                    'edge2': edge1
                })
    
    def _get_relative_position(self, edge1_pos, edge2_pos):
        """
        Determine relative position of piece2 relative to piece1.
        
        Returns: 'top', 'bottom', 'left', 'right', or None
        """
        if edge1_pos == 'top' and edge2_pos == 'bottom':
            return 'top'  # piece2 is above piece1
        elif edge1_pos == 'bottom' and edge2_pos == 'top':
            return 'bottom'  # piece2 is below piece1
        elif edge1_pos == 'left' and edge2_pos == 'right':
            return 'left'  # piece2 is to the left of piece1
        elif edge1_pos == 'right' and edge2_pos == 'left':
            return 'right'  # piece2 is to the right of piece1
        
        return None
    
    def _reverse_direction(self, direction):
        """Get opposite direction."""
        opposites = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }
        return opposites.get(direction)
    
    def _direction_to_offset(self, direction):
        """Convert direction to (row_offset, col_offset)."""
        offsets = {
            'top': (-1, 0),
            'bottom': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        return offsets.get(direction, (0, 0))
    
    def find_corner_pieces(self):
        """
        Find corner pieces (pieces with two border edges).
        
        Returns:
            List of piece_ids that are likely corner pieces
        """
        corners = []
        
        for piece_id, neighbors in self.match_graph.items():
            # Count border edges (edges with no strong matches)
            border_count = 0
            
            # Check all four directions
            directions = {'top', 'bottom', 'left', 'right'}
            matched_dirs = {n['direction'] for n in neighbors if n['score'] < 5.0}
            
            border_count = len(directions - matched_dirs)
            
            if border_count >= 2:
                corners.append(piece_id)
        
        return corners
    
    def find_border_pieces(self):
        """
        Find border pieces (pieces with at least one border edge).
        
        Returns:
            Dictionary mapping piece_id to list of border directions
        """
        borders = {}
        
        for piece_id, neighbors in self.match_graph.items():
            # Check all four directions
            directions = {'top', 'bottom', 'left', 'right'}
            matched_dirs = {n['direction'] for n in neighbors if n['score'] < 5.0}
            
            border_dirs = list(directions - matched_dirs)
            
            if border_dirs:
                borders[piece_id] = border_dirs
        
        return borders
    
    def greedy_assembly(self, start_piece=None, start_position=None):
        """
        Greedy assembly algorithm starting from a corner piece.
        
        Args:
            start_piece: piece_id to start with (if None, auto-select corner)
            start_position: (row, col) position for start piece
            
        Returns:
            Dictionary mapping piece_id to (row, col) position
        """
        # Find starting piece
        if start_piece is None:
            corners = self.find_corner_pieces()
            if corners:
                start_piece = corners[0]
                print(f"Starting from corner piece: {start_piece}")
            else:
                # Just pick first piece
                start_piece = list(self.match_graph.keys())[0]
                print(f"No corners found, starting from: {start_piece}")
        
        # Initialize position
        if start_position is None:
            start_position = (self.grid_size // 2, self.grid_size // 2)
        
        self.piece_positions[start_piece] = start_position
        self.position_pieces[start_position] = start_piece
        
        # Priority queue: (score, piece_id, position)
        queue = []
        
        # Add neighbors of start piece to queue
        for neighbor_info in self.match_graph[start_piece]:
            neighbor = neighbor_info['neighbor']
            direction = neighbor_info['direction']
            score = neighbor_info['score']
            
            # Calculate neighbor position
            row_offset, col_offset = self._direction_to_offset(direction)
            new_row = start_position[0] + row_offset
            new_col = start_position[1] + col_offset
            new_pos = (new_row, new_col)
            
            # Check if position is valid
            if self._is_valid_position(new_pos):
                heapq.heappush(queue, (score, neighbor, new_pos, start_piece))
        
        # Process queue
        max_iterations = self.grid_size * self.grid_size * 10
        iterations = 0
        
        while queue and len(self.piece_positions) < self.grid_size * self.grid_size:
            iterations += 1
            if iterations > max_iterations:
                print("Max iterations reached")
                break
            
            score, piece_id, position, from_piece = heapq.heappop(queue)
            
            # Skip if piece already placed or position occupied
            if piece_id in self.piece_positions:
                continue
            if position in self.position_pieces:
                continue
            
            # Place piece
            self.piece_positions[piece_id] = position
            self.position_pieces[position] = piece_id
            
            print(f"  Placed {piece_id} at {position} (score: {score:.3f}, from: {from_piece})")
            
            # Add neighbors to queue
            for neighbor_info in self.match_graph[piece_id]:
                neighbor = neighbor_info['neighbor']
                direction = neighbor_info['direction']
                n_score = neighbor_info['score']
                
                if neighbor in self.piece_positions:
                    continue
                
                # Calculate neighbor position
                row_offset, col_offset = self._direction_to_offset(direction)
                new_row = position[0] + row_offset
                new_col = position[1] + col_offset
                new_pos = (new_row, new_col)
                
                # Check if position is valid and not occupied
                if self._is_valid_position(new_pos) and new_pos not in self.position_pieces:
                    heapq.heappush(queue, (n_score, neighbor, new_pos, piece_id))
        
        # FORCE PLACEMENT: If pieces are still unplaced, put them in remaining empty spots
        if len(self.piece_positions) < self.grid_size * self.grid_size:
            print(f"\n  Force-placing remaining pieces...")
            
            # Get all piece IDs
            all_pieces = set(self.match_graph.keys())
            placed_pieces = set(self.piece_positions.keys())
            unplaced_pieces = list(all_pieces - placed_pieces)
            
            # Get all empty positions
            all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
            empty_positions = [pos for pos in all_positions if pos not in self.position_pieces]
            
            # Place remaining pieces in empty spots
            for piece_id, position in zip(unplaced_pieces, empty_positions):
                self.piece_positions[piece_id] = position
                self.position_pieces[position] = piece_id
                print(f"  Force-placed {piece_id} at {position}")
        
        print(f"\nAssembly complete: {len(self.piece_positions)}/{self.grid_size*self.grid_size} pieces placed")
        
        return self.piece_positions
    
    def _is_valid_position(self, position):
        """Check if position is within grid bounds."""
        row, col = position
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def layout_pieces_in_grid(self):
        """
        Simple layout: place pieces in order in a grid.
        Useful as a baseline or for debugging.
        
        Returns:
            Dictionary mapping piece_id to (row, col) position
        """
        positions = {}
        piece_ids = sorted(self.match_graph.keys())
        
        for idx, piece_id in enumerate(piece_ids):
            row = idx // self.grid_size
            col = idx % self.grid_size
            
            if row < self.grid_size and col < self.grid_size:
                positions[piece_id] = (row, col)
        
        return positions
    
    def compute_assembly_quality(self, piece_positions):
        """
        Compute quality metrics for the assembly.
        
        Args:
            piece_positions: dict mapping piece_id to (row, col)
            
        Returns:
            Dictionary of quality metrics
        """
        if not piece_positions:
            return {'error': 'No pieces placed'}
        
        total_matches = 0
        matched_edges = 0
        total_edge_score = 0
        
        # Check each placed piece
        for piece_id, position in piece_positions.items():
            row, col = position
            
            # Check all neighbors
            for neighbor_info in self.match_graph[piece_id]:
                neighbor_id = neighbor_info['neighbor']
                direction = neighbor_info['direction']
                score = neighbor_info['score']
                
                # Calculate expected neighbor position
                row_offset, col_offset = self._direction_to_offset(direction)
                expected_pos = (row + row_offset, col + col_offset)
                
                # Check if neighbor is at expected position
                actual_neighbor = piece_positions.get(neighbor_id)
                
                if actual_neighbor == expected_pos:
                    matched_edges += 1
                    total_edge_score += score
                
                total_matches += 1
        
        # Avoid double counting (each edge is counted twice)
        matched_edges = matched_edges // 2
        total_matches = total_matches // 2
        
        quality = {
            'pieces_placed': len(piece_positions),
            'total_pieces': self.grid_size * self.grid_size,
            'completion': len(piece_positions) / (self.grid_size * self.grid_size),
            'matched_edges': matched_edges,
            'total_possible_edges': total_matches,
            'match_accuracy': matched_edges / total_matches if total_matches > 0 else 0,
            'avg_edge_score': total_edge_score / matched_edges if matched_edges > 0 else 0
        }
        
        return quality


def simple_neighbor_matching(all_edges, grid_size):
    """
    Simple assembly based on best matching neighbors.
    Place pieces by finding the best matches iteratively.
    
    Args:
        all_edges: list of dictionaries containing edges for each piece
        grid_size: size of puzzle grid
        
    Returns:
        Dictionary mapping piece_id to (row, col) position
    """
    from edge_matching import find_edge_matches
    
    # Find all matches
    print("Finding edge matches...")
    matches = find_edge_matches(all_edges, compatibility_threshold=10.0, top_k=5)
    
    if not matches:
        print("No matches found, using default grid layout")
        positions = {}
        for idx, piece_edges in enumerate(all_edges):
            piece_id = list(piece_edges.values())[0].piece_id
            row = idx // grid_size
            col = idx % grid_size
            positions[piece_id] = (row, col)
        return positions
    
    # Use PuzzleSolver for assembly
    solver = PuzzleSolver(grid_size, matches)
    positions = solver.greedy_assembly()
    
    # Compute quality
    quality = solver.compute_assembly_quality(positions)
    
    print("\nAssembly Quality Metrics:")
    for key, value in quality.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return positions
