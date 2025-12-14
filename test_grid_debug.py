from image_utils import compute_spatial_energy, evaluate_partition_score
import cv2
import numpy as np

for grid_type in ["2x2", "4x4"]:
    for i in [0, 25]:
        path = f"./Gravity Falls/puzzle_{grid_type}/{i}.jpg"
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        
        energy_x = compute_spatial_energy(gray, 0)
        energy_y = compute_spatial_energy(gray, 1)
        
        max_x = np.max(energy_x)
        max_y = np.max(energy_y)
        
        score_X4 = evaluate_partition_score(energy_x, W, 4)
        score_Y4 = evaluate_partition_score(energy_y, H, 4)
        
        rel_X4 = score_X4 / max_x
        rel_Y4 = score_Y4 / max_y
        
        print(f"{grid_type} img {i}: size={W}x{H}, rel_X4={rel_X4:.3f}, rel_Y4={rel_Y4:.3f}")
