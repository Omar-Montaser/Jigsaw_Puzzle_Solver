"""Test 4x4 solver on 5 images to verify it works."""
import cv2
import numpy as np
from pathlib import Path
from solvers.solver_4x4 import solve_4x4

TEST_IDS = [0, 1, 2, 3, 4]  # First 5 images

def load_pieces(image_path, grid_size=4):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    artifacts = {}
    for idx in range(grid_size * grid_size):
        r, c = idx // grid_size, idx % grid_size
        piece = img[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()
        gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        artifacts[idx] = {
            'rgb': cv2.cvtColor(piece, cv2.COLOR_BGR2RGB),
            'gray': gray,
            'edges': edges,
            'blur': blur
        }
    return artifacts

def reconstruct(artifacts, arrangement, grid_size=4):
    sample = artifacts[0]['rgb']
    ph, pw = sample.shape[:2]
    out = np.zeros((ph*grid_size, pw*grid_size, 3), dtype=np.uint8)
    for idx, pid in enumerate(arrangement):
        r, c = idx // grid_size, idx % grid_size
        out[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = artifacts[pid]['rgb']
    return out

def load_correct(image_id):
    path = Path(f'./Gravity Falls/correct/{image_id}.png')
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def images_match(img1, img2):
    if img1 is None or img2 is None:
        return False
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    f1 = img1.astype(np.float32).flatten()
    f2 = img2.astype(np.float32).flatten()
    f1_norm = f1 - np.mean(f1)
    f2_norm = f2 - np.mean(f2)
    ncc = np.dot(f1_norm, f2_norm) / (np.linalg.norm(f1_norm) * np.linalg.norm(f2_norm) + 1e-10)
    return ncc > 0.95

if __name__ == "__main__":
    print("Testing 4x4 solver on 5 images\n")
    
    correct_count = 0
    for img_id in TEST_IDS:
        puzzle_path = f'./Gravity Falls/puzzle_4x4/{img_id}.jpg'
        if not Path(puzzle_path).exists():
            print(f"✗ Image {img_id}: file not found")
            continue
            
        artifacts = load_pieces(puzzle_path)
        board, arrangement, score = solve_4x4(artifacts, verbose=False)
        solved = reconstruct(artifacts, arrangement)
        correct = load_correct(img_id)
        match = images_match(solved, correct)
        
        if match:
            correct_count += 1
        
        status = "✓" if match else "✗"
        print(f"{status} Image {img_id}: score={score:.2f}")
    
    print(f"\nResult: {correct_count}/{len(TEST_IDS)} correct")
