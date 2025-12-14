from image_utils import detect_grid_size

print("4x4 images:")
for i in [0, 10, 25, 30, 50]:
    path = f"./Gravity Falls/puzzle_4x4/{i}.jpg"
    grid = detect_grid_size(path)
    status = "OK" if grid == 4 else "WRONG"
    print(f"  Image {i}: {grid}x{grid} {status}")

print("\n2x2 images:")
for i in [0, 10, 25, 30, 50]:
    path = f"./Gravity Falls/puzzle_2x2/{i}.jpg"
    grid = detect_grid_size(path)
    status = "OK" if grid == 2 else "WRONG"
    print(f"  Image {i}: {grid}x{grid} {status}")
