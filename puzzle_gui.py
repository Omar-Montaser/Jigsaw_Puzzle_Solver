#!/usr/bin/env python
"""
Jigsaw Puzzle Solver GUI

A game-styled Tkinter GUI for the puzzle solver.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import threading
import sys
import os


# Color scheme
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent': '#e94560',
    'accent_hover': '#ff6b6b',
    'text': '#eaeaea',
    'text_dim': '#a0a0a0',
    'success': '#4ecca3',
    'error': '#ff6b6b',
}


class TextRedirector:
    """Redirects stdout/stderr to a Tkinter text widget."""
    def __init__(self, widget, root):
        self.widget = widget
        self.root = root

    def write(self, text):
        self.root.after(0, lambda: self._write(text))

    def _write(self, text):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass


class PuzzleSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 Jigsaw Puzzle Solver")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)
        self.root.configure(bg=COLORS['bg_dark'])
        
        self.image_path = None
        self.solved_image = None
        self.photo = None
        
        self._setup_styles()
        self._build_ui()
    
    def _setup_styles(self):
        """Configure ttk styles for game-like appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame styles
        style.configure('Dark.TFrame', background=COLORS['bg_dark'])
        style.configure('Card.TFrame', background=COLORS['bg_medium'])
        
        # Button styles
        style.configure('Game.TButton',
                        font=('Segoe UI', 11, 'bold'),
                        padding=(20, 12),
                        background=COLORS['accent'],
                        foreground='white')
        style.map('Game.TButton',
                  background=[('active', COLORS['accent_hover']),
                              ('disabled', '#555555')])
        
        style.configure('Secondary.TButton',
                        font=('Segoe UI', 10),
                        padding=(15, 10),
                        background=COLORS['bg_light'],
                        foreground='white')
        style.map('Secondary.TButton',
                  background=[('active', COLORS['bg_medium'])])
        
        # Label styles
        style.configure('Title.TLabel',
                        font=('Segoe UI', 24, 'bold'),
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text'])
        
        style.configure('Subtitle.TLabel',
                        font=('Segoe UI', 11),
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_dim'])
        
        style.configure('File.TLabel',
                        font=('Segoe UI', 10),
                        background=COLORS['bg_dark'],
                        foreground=COLORS['accent'])
        
        style.configure('Status.TLabel',
                        font=('Segoe UI', 10),
                        background=COLORS['bg_medium'],
                        foreground=COLORS['text'],
                        padding=(10, 8))
        
        # Progress bar
        style.configure('Game.Horizontal.TProgressbar',
                        background=COLORS['accent'],
                        troughcolor=COLORS['bg_light'],
                        thickness=6)
        
        # LabelFrame
        style.configure('Card.TLabelframe',
                        background=COLORS['bg_medium'],
                        foreground=COLORS['text'])
        style.configure('Card.TLabelframe.Label',
                        font=('Segoe UI', 10, 'bold'),
                        background=COLORS['bg_medium'],
                        foreground=COLORS['text'])
    
    def _build_ui(self):
        # Main container
        main = ttk.Frame(self.root, style='Dark.TFrame', padding=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(main, style='Dark.TFrame')
        header.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header, text="🧩 Puzzle Solver", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header, text="Select an image and watch it get solved!",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0), pady=(8, 0))
        
        # Controls row
        ctrl = ttk.Frame(main, style='Dark.TFrame')
        ctrl.pack(fill=tk.X, pady=(0, 15))
        
        self.select_btn = ttk.Button(ctrl, text="📁 Select Image",
                                      style='Secondary.TButton', command=self._select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.solve_btn = ttk.Button(ctrl, text="▶ Solve Puzzle",
                                     style='Game.TButton', command=self._solve, state=tk.DISABLED)
        self.solve_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_btn = ttk.Button(ctrl, text="↺ Reset",
                                     style='Secondary.TButton', command=self._reset)
        self.reset_btn.pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(ctrl, text="No file selected", style='File.TLabel')
        self.file_label.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main, style='Game.Horizontal.TProgressbar',
                                         mode='indeterminate', length=400)
        self.progress.pack(fill=tk.X, pady=(0, 15))
        
        # Content: Image (large) + Log (small)
        content = ttk.Frame(main, style='Dark.TFrame')
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=3)  # Image gets more space
        content.columnconfigure(1, weight=1)  # Log gets less space
        content.rowconfigure(0, weight=1)
        
        # Left: Result image (larger)
        img_frame = tk.Frame(content, bg=COLORS['bg_medium'], bd=0, highlightthickness=2,
                             highlightbackground=COLORS['bg_light'])
        img_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        img_header = tk.Frame(img_frame, bg=COLORS['bg_medium'])
        img_header.pack(fill=tk.X, padx=15, pady=(15, 10))
        tk.Label(img_header, text="🖼 Result", font=('Segoe UI', 12, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text']).pack(side=tk.LEFT)
        
        self.img_container = tk.Frame(img_frame, bg=COLORS['bg_dark'])
        self.img_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.img_label = tk.Label(self.img_container, text="Solved puzzle will appear here",
                                   font=('Segoe UI', 11), bg=COLORS['bg_dark'],
                                   fg=COLORS['text_dim'])
        self.img_label.pack(expand=True)
        
        # Right: Log (smaller)
        log_frame = tk.Frame(content, bg=COLORS['bg_medium'], bd=0, highlightthickness=2,
                             highlightbackground=COLORS['bg_light'])
        log_frame.grid(row=0, column=1, sticky='nsew')
        
        log_header = tk.Frame(log_frame, bg=COLORS['bg_medium'])
        log_header.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(log_header, text="📋 Log", font=('Segoe UI', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text']).pack(side=tk.LEFT)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, state='disabled',
            font=('Consolas', 8), height=10, width=30,
            bg=COLORS['bg_dark'], fg=COLORS['text'],
            insertbackground=COLORS['text'],
            selectbackground=COLORS['accent'],
            bd=0, highlightthickness=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Status bar
        status_frame = tk.Frame(main, bg=COLORS['bg_medium'], height=40)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready to solve puzzles!",
                                      font=('Segoe UI', 10), bg=COLORS['bg_medium'],
                                      fg=COLORS['text'], anchor=tk.W, padx=15)
        self.status_label.pack(fill=tk.BOTH, expand=True)
    
    def _log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
    
    def _set_status(self, msg, color=None):
        self.status_label.config(text=msg, fg=color or COLORS['text'])
    
    def _select_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Puzzle Image", filetypes=filetypes)
        if path:
            self.image_path = path
            filename = os.path.basename(path)
            self.file_label.config(text=f"📄 {filename}")
            self.solve_btn.config(state=tk.NORMAL)
            self._log(f"Selected: {filename}")
            self._set_status(f"Image loaded: {filename}")
    
    def _solve(self):
        if not self.image_path:
            return
        
        self.select_btn.config(state=tk.DISABLED)
        self.solve_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self._set_status("🔄 Solving puzzle...", COLORS['accent'])
        
        self.img_label.config(image='', text="⏳ Solving...", fg=COLORS['accent'])
        self.photo = None
        
        thread = threading.Thread(target=self._run_solver, daemon=True)
        thread.start()
    
    def _find_correct_image(self, puzzle_path):
        """Try to find the corresponding correct image for accuracy calculation."""
        from pathlib import Path
        puzzle_path = Path(puzzle_path)
        
        # Try common patterns: puzzle in puzzle_NxN folder, correct in correct folder
        # e.g., "Gravity Falls/puzzle_8x8/0.jpg" -> "Gravity Falls/correct/0.png"
        parent = puzzle_path.parent
        filename_stem = puzzle_path.stem
        
        # Check if parent folder matches puzzle_NxN pattern
        if parent.name.startswith('puzzle_'):
            correct_folder = parent.parent / 'correct'
            if correct_folder.exists():
                # Try different extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    correct_path = correct_folder / (filename_stem + ext)
                    if correct_path.exists():
                        return str(correct_path)
        
        return None

    def _run_solver(self):
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = TextRedirector(self.log_text, self.root)
            sys.stderr = TextRedirector(self.log_text, self.root)
            
            from pipeline import solve_and_reconstruct
            arrangement, score, solved_img = solve_and_reconstruct(self.image_path, verbose=True)
            
            # Try to compute accuracy if correct image exists
            accuracy = None
            correct_path = self._find_correct_image(self.image_path)
            if correct_path:
                try:
                    from accuracy_utils import (
                        load_ground_truth,
                        compute_pairwise_neighbor_accuracy,
                        arrangement_to_grid,
                    )
                    # Detect grid size from arrangement length
                    import math
                    grid_size = int(math.sqrt(len(arrangement)))
                    
                    gt_labels = load_ground_truth(self.image_path, correct_path, grid_size)
                    pred_grid = arrangement_to_grid(arrangement, grid_size)
                    accuracy = compute_pairwise_neighbor_accuracy(pred_grid, gt_labels)
                    print(f"\nAccuracy: {accuracy:.2%}")
                except Exception as e:
                    print(f"\nCould not compute accuracy: {e}")
            
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.solved_image = solved_img
            self.root.after(0, lambda: self._on_solve_complete(score, accuracy))
            
        except Exception as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.root.after(0, lambda: self._on_solve_error(str(e)))
    
    def _on_solve_complete(self, score, accuracy=None):
        self.progress.stop()
        self.select_btn.config(state=tk.NORMAL)
        self.solve_btn.config(state=tk.NORMAL)
        
        if self.solved_image is not None:
            self._display_image(self.solved_image)
        
        # Display accuracy if available, otherwise fall back to score
        if accuracy is not None:
            self._log(f"\nAccuracy: {accuracy:.2%}")
            if accuracy >= 0.95:
                self._log("✅ Puzzle Solved!")
                self._set_status(f"✅ Puzzle Solved! (Accuracy: {accuracy:.2%})", COLORS['success'])
            elif accuracy >= 0.70:
                self._log("⚠ Partially solved")
                self._set_status(f"⚠ Partially Solved (Accuracy: {accuracy:.2%})", COLORS['accent'])
            else:
                self._log("⚠ May need manual review")
                self._set_status(f"⚠ Low Accuracy ({accuracy:.2%})", COLORS['error'])
        else:
            # No correct image found, show score only
            self._log(f"\nScore: {score:.4f}")
            self._log("ℹ No correct image found for accuracy")
            self._set_status(f"Completed (Score: {score:.4f})", COLORS['text_dim'])
    
    def _on_solve_error(self, error_msg):
        self.progress.stop()
        self.select_btn.config(state=tk.NORMAL)
        self.solve_btn.config(state=tk.NORMAL)
        
        self._log(f"\n❌ Error: {error_msg}")
        self._set_status("❌ Error occurred", COLORS['error'])
        self.img_label.config(text="❌ Error", fg=COLORS['error'])
    
    def _display_image(self, cv_img):
        import cv2
        
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        self.img_container.update_idletasks()
        label_w = max(self.img_container.winfo_width() - 20, 200)
        label_h = max(self.img_container.winfo_height() - 20, 200)
        
        img_w, img_h = pil_img.size
        ratio = min(label_w / img_w, label_h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.photo, text='', bg=COLORS['bg_dark'])
    
    def _reset(self):
        self.image_path = None
        self.solved_image = None
        self.photo = None
        
        self.file_label.config(text="No file selected")
        self.solve_btn.config(state=tk.DISABLED)
        self.img_label.config(image='', text="Solved puzzle will appear here",
                               fg=COLORS['text_dim'])
        
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
        self._set_status("Ready to solve puzzles!")
        self.progress.stop()


def main():
    root = tk.Tk()
    app = PuzzleSolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()