"""Image splitting into puzzle pieces."""

import numpy as np


def split_image(image_data, grid_size):
    """
    Split image into grid_size x grid_size patches.
    
    Args:
        image_data: Input image as numpy array
        grid_size: Number of divisions per dimension
    
    Returns:
        List of image patches in row-major order
    """
    if image_data is None:
        return []
        
    height, width = image_data.shape[:2]
    patch_H = height // grid_size
    patch_W = width // grid_size
    
    sections = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * patch_H
            y_end = (i + 1) * patch_H
            x_start = j * patch_W
            x_end = (j + 1) * patch_W
            
            section = image_data[y_start:y_end, x_start:x_end].copy()
            sections.append(section)
            
    return sections


def split_image_to_dict(image_data, grid_size):
    """
    Split image into patches and return as dictionary.
    
    Args:
        image_data: Input image as numpy array
        grid_size: Number of divisions per dimension
    
    Returns:
        Dict mapping piece_id (int) to image patch
    """
    patches = split_image(image_data, grid_size)
    return {idx: patch for idx, patch in enumerate(patches)}
