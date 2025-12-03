import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path):
    """Load image and convert to numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        return None

def detect_grid_size(image_path):
    """Detect puzzle grid size (2x2, 4x4, or 8x8) using gradient analysis."""
    img = load_image(image_path)
    if img is None:
        return None
        
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    h, w = gray.shape
    
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    
    row_profile = np.mean(grad_y, axis=1)
    col_profile = np.mean(grad_x, axis=0)
    
    def check_cuts(profile, fractions, threshold_ratio=1.2):
        length = len(profile)
        scores = []
        baseline = np.mean(profile)
        
        for frac in fractions:
            # The cut is expected at index: length * frac - 1
            idx = int(length * frac) - 1
            
            # Check a small window around the expected cut
            start = max(0, idx - 2)
            end = min(length, idx + 3)
            
            if start >= end:
                scores.append(0)
                continue
                
            peak_val = np.max(profile[start:end])
            scores.append(peak_val / (baseline + 1e-6))
            
        return np.mean(scores)
    
    # Threshold for detecting a grid line
    # Shuffled images usually have strong edges at cuts.
    THRESHOLD = 1.5
    
    # Check 8x8 specific cuts (1/8, 3/8, 5/8, 7/8)
    score_8_y = check_cuts(row_profile, [1/8, 3/8, 5/8, 7/8])
    score_8_x = check_cuts(col_profile, [1/8, 3/8, 5/8, 7/8])
    
    if score_8_y > THRESHOLD and score_8_x > THRESHOLD:
        return 8
        
    # Check 4x4 specific cuts (1/4, 3/4)
    score_4_y = check_cuts(row_profile, [1/4, 3/4])
    score_4_x = check_cuts(col_profile, [1/4, 3/4])
    
    if score_4_y > THRESHOLD and score_4_x > THRESHOLD:
        return 4
        
    # Check 2x2 specific cuts (1/2)
    score_2_y = check_cuts(row_profile, [1/2])
    score_2_x = check_cuts(col_profile, [1/2])
    
    if score_2_y > THRESHOLD and score_2_x > THRESHOLD:
        return 2
        
    return None

def split_image(image_array, grid_size):
    """Split image into grid_size x grid_size patches."""
    if image_array is None:
        return []
        
    height, width = image_array.shape[:2]
    patch_height = height // grid_size
    patch_width = width // grid_size
    
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * patch_height
            y_end = (i + 1) * patch_height
            x_start = j * patch_width
            x_end = (j + 1) * patch_width
            
            patch = image_array[y_start:y_end, x_start:x_end]
            patches.append(patch)
            
    return patches

def display_patches_with_indices(patches, grid_size):
    if not patches:
        print("No patches to display.")
        return

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, ax in enumerate(axes):
        if idx < len(patches):
            ax.imshow(patches[idx])
            ax.set_title(f"Index: {idx}")
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()
