import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def load_image(file_path):
    """Load image from path and return as numpy array (RGB)."""
    try:
        # Load using PIL for standard image handling in notebooks
        pic = Image.open(file_path)
        return np.array(pic)
    except Exception as e:
        print(f"File handling error: {e}")
        return None

def compute_spatial_energy(image_grayscale, axis_orientation):
    # Reduce noise with a mild blur
    img_smooth = cv2.GaussianBlur(image_grayscale, (3, 3), 0)

    if axis_orientation == 0: 
        # Detect vertical boundaries (dx=1, dy=0)
        grad_map = cv2.Sobel(img_smooth, cv2.CV_64F, 1, 0, ksize=3)
        # Sum absolute gradients along the vertical axis (axis=0) to get the horizontal profile
        profile = np.sum(np.abs(grad_map), axis=0)
        
    else:
        # Detect horizontal boundaries (dx=0, dy=1)
        grad_map = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
        # Sum absolute gradients along the horizontal axis (axis=1) to get the vertical profile
        profile = np.sum(np.abs(grad_map), axis=1)
    
    return profile

def evaluate_partition_score(energy_profile, total_length, hypo_N):
    slice_size = total_length / hypo_N    
    # Check only odd-numbered slices (1/N, 3/N, 5/N...)
    # This prevents counting the central 1/2 line as evidence for 4x4 or 8x8.
    check_points = [int(slice_size * i) for i in range(1, hypo_N, 2)]

    peak_strengths = []
    for center_index in check_points:
        if center_index >= len(energy_profile): continue
        
        # Look for the peak in a 9-pixel window around the expected center
        start_idx = max(0, center_index - 4) 
        end_idx = min(len(energy_profile), center_index + 5)
        
        if start_idx >= end_idx: continue
        
        max_energy = np.max(energy_profile[start_idx:end_idx])
        peak_strengths.append(max_energy)
    
    if not peak_strengths:
        return 0.0

    return np.mean(peak_strengths)

def detect_grid_size(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error reading image: {image_path}")
        return None
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    # 1. Compute Energy Profiles
    # energy_x: Profile of vertical lines, mapped to the X-axis (width)
    # energy_y: Profile of horizontal lines, mapped to the Y-axis (height)
    energy_x = compute_spatial_energy(img_gray, axis_orientation=0) 
    energy_y = compute_spatial_energy(img_gray, axis_orientation=1)

    # 2. Find maximum global energy (typically the strong central cut)
    max_energy_x = np.max(energy_x)
    max_energy_y = np.max(energy_y)

    # Prevent division by zero if image is blank
    max_energy_x = max_energy_x if max_energy_x > 0 else 1.0
    max_energy_y = max_energy_y if max_energy_y > 0 else 1.0

    # 3. Evaluate Hypotheses
    # Score 4: Mean energy of cuts at 1/4 and 3/4
    score_X4 = evaluate_partition_score(energy_x, W, 4)
    score_Y4 = evaluate_partition_score(energy_y, H, 4)
  
    # Score 8: Mean energy of cuts at 1/8, 3/8, 5/8, 7/8
    score_X8 = evaluate_partition_score(energy_x, W, 8)
    score_Y8 = evaluate_partition_score(energy_y, H, 8)
   
    # 4. Hierarchical Decision using Relative Strength (The Fix)
    # The new lines must be at least X% as strong as the max line to confirm the finer grid.
    REL_STRENGTH_8 = 0.52  
    REL_STRENGTH_4 = 0.53 

    # Check 8x8
    confirmed_X8 = (score_X8 / max_energy_x) > REL_STRENGTH_8
    confirmed_Y8 = (score_Y8 / max_energy_y) > REL_STRENGTH_8

    if confirmed_X8 and confirmed_Y8:
        return 8
    
    # Check 4x4
    confirmed_X4 = (score_X4 / max_energy_x) > REL_STRENGTH_4
    confirmed_Y4 = (score_Y4 / max_energy_y) > REL_STRENGTH_4
        
    if confirmed_X4 and confirmed_Y4:
        return 4
    
    # Default to 2x2 if neither finer grid is confirmed
    return 2

def split_image(image_data, dimension):
    """Splits image data into dimension x dimension patches."""
    if image_data is None:
        return []
        
    height, width = image_data.shape[:2]
    patch_H = height // dimension
    patch_W = width // dimension
    
    sections = []
    for i in range(dimension):
        for j in range(dimension):
            y_start = i * patch_H
            y_end = (i + 1) * patch_H
            x_start = j * patch_W
            x_end = (j + 1) * patch_W
            
            section = image_data[y_start:y_end, x_start:x_end]
            sections.append(section)
            
    return sections

def display_patches_with_indices(sections, dimension):
    """Displays the list of image sections in a grid format with indices."""
    if not sections:
        print("No sections to display.")
        return

    fig, axes = plt.subplots(dimension, dimension, figsize=(8, 8))
    
    # Handle subplot arrangement for 1x1, 2x2, 4x4, 8x8 cases
    if dimension == 1:
        axes = np.array([[axes]])
    elif not isinstance(axes, np.ndarray) or axes.ndim == 1:
         axes = np.array(axes).reshape(dimension, dimension)
        
    for idx, patch in enumerate(sections):
        row = idx // dimension
        col = idx % dimension
        
        ax = axes[row, col]
        ax.imshow(patch)
        ax.axis('off')
        ax.set_title(f"Section {idx}", fontsize=8)
        
    plt.tight_layout()
    plt.show()