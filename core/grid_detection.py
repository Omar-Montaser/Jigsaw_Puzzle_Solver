"""Grid size detection for puzzle images."""

import cv2
import numpy as np


def compute_spatial_energy(image_grayscale, axis_orientation):
    """
    Compute gradient energy profile along an axis.
    
    Args:
        image_grayscale: Grayscale image
        axis_orientation: 0 for vertical boundaries, 1 for horizontal
    
    Returns:
        1D energy profile
    """
    img_smooth = cv2.GaussianBlur(image_grayscale, (3, 3), 0)

    if axis_orientation == 0: 
        grad_map = cv2.Sobel(img_smooth, cv2.CV_64F, 1, 0, ksize=3)
        profile = np.sum(np.abs(grad_map), axis=0)
    else:
        grad_map = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
        profile = np.sum(np.abs(grad_map), axis=1)
    
    return profile


def evaluate_partition_score(energy_profile, total_length, hypo_N):
    """
    Evaluate how well the energy profile matches a hypothesized grid size.
    
    Args:
        energy_profile: 1D energy profile
        total_length: Total length of the dimension
        hypo_N: Hypothesized number of divisions
    
    Returns:
        Mean peak strength at expected cut locations
    """
    slice_size = total_length / hypo_N    
    check_points = [int(slice_size * i) for i in range(1, hypo_N, 2)]

    peak_strengths = []
    for center_index in check_points:
        if center_index >= len(energy_profile):
            continue
        
        start_idx = max(0, center_index - 4) 
        end_idx = min(len(energy_profile), center_index + 5)
        
        if start_idx >= end_idx:
            continue
        
        max_energy = np.max(energy_profile[start_idx:end_idx])
        peak_strengths.append(max_energy)
    
    if not peak_strengths:
        return 0.0

    return np.mean(peak_strengths)


def detect_grid_size(image_path):
    """
    Detect the grid size of a puzzle image.
    
    Args:
        image_path: Path to the puzzle image
    
    Returns:
        Grid size (2, 4, or 8)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error reading image: {image_path}")
        return None
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    energy_x = compute_spatial_energy(img_gray, axis_orientation=0) 
    energy_y = compute_spatial_energy(img_gray, axis_orientation=1)

    max_energy_x = np.max(energy_x) if np.max(energy_x) > 0 else 1.0
    max_energy_y = np.max(energy_y) if np.max(energy_y) > 0 else 1.0

    score_X4 = evaluate_partition_score(energy_x, W, 4)
    score_Y4 = evaluate_partition_score(energy_y, H, 4)
    score_X8 = evaluate_partition_score(energy_x, W, 8)
    score_Y8 = evaluate_partition_score(energy_y, H, 8)
   
    REL_STRENGTH_8 = 0.52  
    REL_STRENGTH_4 = 0.53 

    confirmed_X8 = (score_X8 / max_energy_x) > REL_STRENGTH_8
    confirmed_Y8 = (score_Y8 / max_energy_y) > REL_STRENGTH_8

    if confirmed_X8 and confirmed_Y8:
        return 8
    
    confirmed_X4 = (score_X4 / max_energy_x) > REL_STRENGTH_4
    confirmed_Y4 = (score_Y4 / max_energy_y) > REL_STRENGTH_4
        
    if confirmed_X4 and confirmed_Y4:
        return 4
    
    return 2
