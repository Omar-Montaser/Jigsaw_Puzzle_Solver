# Jigsaw Puzzle Solver: Automated Assembly Using Classical Computer Vision

## 📌 Project Overview
This project implements a complete **Jigsaw Puzzle Solver** using classical computer vision techniques (no machine learning or AI). The system processes shuffled puzzle images, extracts individual pieces, analyzes their edge shapes, and automatically assembles them by finding complementary edge matches.

**Project Components:**
- **Phase 1**: Preprocessing and edge detection of puzzle pieces
- **Phase 2**: Edge matching and puzzle assembly


## 📂 Project Structure

### Phase 1 Files:
* **`phase1_pipeline.ipynb`** - Main preprocessing pipeline notebook
* **`image_utils.py`** - Core functions for grid detection, image splitting, and utilities
* **`edge_detection_visualizer.ipynb`** - Demo of edge detection filters
* **`grid_detection_visualizer.ipynb`** - Debug visualization for grid detection
* **`imageSplit_visualizer.ipynb`** - Piece splitting demonstration

### Phase 2 Files:
* **`phase2_pipeline.ipynb`** - Main edge matching and assembly pipeline
* **`edge_matching.py`** - Edge extraction, shape descriptors, and matching algorithms
* **`puzzle_assembly.py`** - Puzzle assembly logic using graph-based greedy approach
* **`puzzle_visualization.py`** - Visualization tools for matches and assembly

### Data Directories:
* **`Gravity Falls/`** - Input puzzle images (2x2, 4x4, 8x8 grids)
* **`processed_artifacts/`** - Phase 1 outputs (edge images, enhanced pieces, metadata)
* **`phase2_results/`** - Phase 2 outputs (match reports, assembly results)



## ⚙️ Phase 1: Preprocessing and Edge Detection

### Key Features:
1. **Automatic Grid Detection:** Uses Sobel gradient energy profiles with relative strength thresholds to identify puzzle grid size (2x2, 4x4, or 8x8). The algorithm checks ODD-numbered divisions (1/N, 3/N, 5/N...) to avoid false positives.
2. **Adaptive Preprocessing:** Implements **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to handle low-contrast images.
3. **Noise-Resistant Edge Detection:** Utilizes **Canny Edge Detection** & **Bilateral Filtering** to ignore internal puzzle textures while preserving piece boundaries.

### Processing Pipeline:
1. Load puzzle images from dataset
2. Detect grid size using energy-based analysis
3. Split images into individual pieces
4. Apply enhancement pipeline:
   - CLAHE for contrast enhancement
   - Bilateral filter for noise reduction
   - Gaussian blur for smoothing
   - Canny edge detection
   - Morphological closing to connect edges
5. Save processed artifacts (enhanced pieces, edge images, metadata)

### Phase 1 Outputs:
- **Enhanced pieces**: Contrast-enhanced grayscale pieces
- **Edge images**: Binary edge maps from Canny detection
- **Metadata**: Processing parameters and piece information
- **Summary**: Overall processing statistics

---

## 🧩 Phase 2: Edge Matching and Assembly

### Key Features:
1. **Edge Representation:** Rotation-invariant shape descriptors for each puzzle piece edge
2. **Shape Descriptors:**
   - **Fourier Descriptors**: Frequency-domain representation (rotation-invariant)
   - **Curvature Signatures**: Local curvature along edge contour
   - **Centroid Distance Signatures**: Distance from edge points to centroid
   - **Straightness Metric**: For detecting flat border edges
3. **Edge Classification:** Automatic detection of border (flat) vs. internal (interlocking) edges
4. **Complementary Matching:** Finds edges that fit together like lock and key
5. **Assembly Algorithm:** Greedy approach with priority queue, starting from corner pieces

### Assembly Pipeline:
1. Load Phase 1 edge images
2. Extract contours and identify 4 edges per piece (top, bottom, left, right)
3. Compute shape descriptors for each edge
4. Compare all edge pairs using similarity metrics
5. Find best matching complementary edges
6. Identify corner and border pieces
7. Assemble puzzle using greedy algorithm:
   - Start from corner piece
   - Add neighbors with highest confidence matches
   - Continue until all pieces placed or no more matches
8. Visualize results and compute quality metrics

### Phase 2 Techniques:

#### 1. Edge Extraction
- Use `cv2.findContours()` to extract piece boundaries
- Divide contours into 4 edges based on bounding box regions
- Filter edges by minimum length to remove noise

#### 2. Shape Descriptors
**Fourier Descriptors:**
```python
# Convert contour to complex representation
complex_points = x + 1j*y
fft_result = np.fft.fft(complex_points)
# Use magnitude for rotation invariance
descriptors = np.abs(fft_result[1:N]) / abs(fft_result[1])
```

**Curvature Signature:**
- Compute local curvature using cross product of consecutive vectors
- Provides complementary pattern (positive curvature matches negative)

**Centroid Distance:**
- Distance from each edge point to edge centroid
- Normalized for scale invariance

#### 3. Edge Matching
**Similarity Metric:**
$$\text{similarity} = w_1 \|F_1 - F_2\| + w_2 \|C_1 + C_2\| + w_3 \|D_1 - D_2^{rev}\|$$

Where:
- $F$ = Fourier descriptors
- $C$ = Curvature signature (negated for complementary matching)
- $D$ = Centroid distances (reversed for complementary matching)
- $w_i$ = weights (default: 1.0, 0.8, 0.6)

**Compatibility Rules:**
- Top edges match with bottom edges
- Left edges match with right edges
- Border edges only match with border edges
- Lower score = better match

#### 4. Assembly Algorithm
**Greedy Assembly with Priority Queue:**
1. Find corner pieces (2 border edges)
2. Start from corner, place at center position
3. Add all neighbors to priority queue (sorted by match score)
4. Pop best match from queue:
   - Check if piece already placed → skip
   - Check if position occupied → skip
   - Place piece and add its neighbors to queue
5. Repeat until puzzle complete or queue empty

**Quality Metrics:**
- Completion percentage (pieces placed / total pieces)
- Match accuracy (correct neighbor placements)
- Average edge match score

### Phase 2 Outputs:
- **Match reports**: JSON files with top N matches and scores
- **Assembly results**: Piece positions and quality metrics
- **Visualizations**: 
  - Side-by-side edge comparisons
  - Match graph with connections
  - Assembled puzzle image
  - Shape descriptor plots (for debugging)

---

## 🎯 Design Decisions and Justifications

### Why Fourier Descriptors?
- **Rotation invariant** by using magnitude only
- **Compact representation** (20 coefficients capture shape)
- **Well-established** in shape matching literature
- Captures global shape characteristics

### Why Curvature Signatures?
- Captures local **"lock and key"** complementary shapes
- Interlocking edges have **opposite curvature patterns**
- More discriminative than global descriptors alone

### Why Greedy Assembly?
- **Efficient**: O(N²log N) for N pieces
- **Works well** for small-medium puzzles (2x2, 4x4)
- **Interpretable**: can visualize decision process
- Foundation for more sophisticated methods (backtracking, global optimization)

### Limitations:
1. **No rotation handling**: Assumes pieces are upright in captured images
2. **Local decisions**: Greedy approach can't backtrack from mistakes
3. **Scale sensitive**: Pieces must be similar size
4. **Texture ignored**: Only uses edge shape, not piece content
5. **Border detection**: Simple straightness metric may miss slight curves

---

## 🚀 How to Run

### Phase 1: Preprocessing
```python
# Run the main pipeline notebook
jupyter notebook phase1_pipeline.ipynb

# Or run programmatically:
from image_utils import detect_grid_size, split_image, load_image
from phase1_pipeline import process_puzzle_image

result = process_puzzle_image(
    "Gravity Falls/puzzle_4x4/6.jpg",
    output_dir="processed_artifacts",
    save_artifacts=True,
    visualize=True
)
```

### Phase 2: Assembly
```python
# Run the assembly pipeline notebook
jupyter notebook phase2_pipeline.ipynb

# Or run programmatically:
from edge_matching import extract_all_edges_from_puzzle, find_edge_matches
from puzzle_assembly import PuzzleSolver
from puzzle_visualization import visualize_assembly_result

# Extract edges
all_edges = extract_all_edges_from_puzzle("processed_artifacts", "puzzle_2x2", "0")

# Find matches
matches = find_edge_matches(all_edges, compatibility_threshold=10.0, top_k=5)

# Assemble
solver = PuzzleSolver(grid_size=2, matches=matches)
positions = solver.greedy_assembly()

# Visualize
visualize_assembly_result(positions, piece_images, grid_size=2)
```

---

## 📊 Results and Performance

### Phase 1 Statistics:
- **Total images processed**: 330 (110 each for 2x2, 4x4, 8x8)
- **Grid detection accuracy**: ~95% (manually verified on sample)
- **Average processing time**: ~2-3 seconds per image

### Phase 2 Performance (Actual Results):

**⚠️ Important: The system demonstrates classical CV capabilities AND limitations**

- **2x2 Puzzles**: Variable results (25-50% accuracy on tested images)
- **4x4 Puzzles**: Partial assembly expected (testing in progress)
- **8x8 Puzzles**: Significant challenges expected
- **Border detection**: Works with adjusted threshold (0.65 vs 0.8)

### Actual Test Results:
| Image ID | Pieces Placed | Match Accuracy | Best Match Score | Notes |
|----------|---------------|----------------|------------------|-------|
| 2x2_0    | 2/4 (50%)     | 50%            | 3.255           | Improved border detection |
| 2x2_1    | 1/4 (25%)     | 0%             | 3.912           | Poor edge extraction |
| 2x2_5    | 2/4 (50%)     | 25%            | 1.850           | Best score but still wrong |

### Key Findings:
**What Works:**
- ✅ Edge extraction and contour detection
- ✅ Shape descriptor computation (Fourier, curvature)
- ✅ Rotation-invariant matching framework
- ✅ Border piece identification (with tuned threshold)
- ✅ Systematic assembly algorithm

**Critical Limitations:**
- ❌ **Shape alone is insufficient** - edges with different textures can have similar shapes
- ❌ **No texture/content matching** - algorithm is "blind" to image content
- ❌ **No error correction** - greedy algorithm can't backtrack from mistakes
- ❌ **Dataset-specific challenges** - Gravity Falls images have complex textures that interfere with edge detection

---

## 📚 References and Resources

### Academic Papers:
1. **Fourier Descriptors**:
   - Zahn, C.T., Roskies, R.Z. (1972). "Fourier Descriptors for Plane Closed Curves". *IEEE Transactions on Computers*.
   
2. **Shape Matching**:
   - Belongie, S., Malik, J., Puzicha, J. (2002). "Shape Matching and Object Recognition Using Shape Contexts". *IEEE TPAMI*.

3. **Jigsaw Puzzle Assembly**:
   - Pomeranz, D., Shemesh, M., Ben-Shahar, O. (2011). "A Fully Automated Greedy Square Jigsaw Puzzle Solver". *CVPR*.
   - Cho, T.S., Avidan, S., Freeman, W.T. (2010). "A Probabilistic Image Jigsaw Puzzle Solver". *CVPR*.

### OpenCV Documentation:
- Edge Detection: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- Contour Features: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- Fourier Transform: https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html

### Techniques Learned From:
- CLAHE: Adaptive histogram equalization for low-contrast images
- Bilateral Filtering: Edge-preserving smoothing
- Morphological Operations: Connecting broken edges
- Shape Descriptors: Rotation-invariant representations
- Graph-based Assembly: Priority queue for greedy matching

---

## 🔮 Future Improvements

### Short-term:
1. **Rotation handling**: Add rotation estimation and matching
2. **Better scoring**: Incorporate texture similarity alongside shape
3. **Backtracking**: Allow algorithm to undo incorrect placements
4. **Parameter tuning**: Optimize thresholds for different puzzle types

### Long-term:
1. **Global optimization**: Use simulated annealing or genetic algorithms
2. **Piece orientation**: Detect and correct rotated pieces
3. **Broken pieces**: Handle damaged or irregular pieces
4. **Multiple puzzles**: Separate and solve mixed puzzle sets
5. **Real-time solving**: Live camera feed with continuous assembly

---

## 👥 Course Information
**Course**: CSE483 / CESS5004 – Computer Vision  
**Project**: Jigsaw Puzzle Solver using Classical CV Techniques  
**Milestone 1**: Preprocessing and Edge Detection ✅  
**Milestone 2**: Edge Matching and Assembly 🚧

---

## 📝 License and Usage
This project is for educational purposes as part of the Computer Vision course. The techniques demonstrated use only classical computer vision methods without machine learning or AI, as required by the project specification.
