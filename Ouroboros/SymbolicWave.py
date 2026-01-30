"""
SymbolicWave.py — Triangulation & Waveform Encoder for Vast Numbers
January 16, 2026

Updated for full reversibility:
- Dynamic padding in all directions (top/left/bottom/right) with margin
- Exact line drawing without clipping or bounds errors
- Proper Bresenham implementation with clear row/col semantics
- Added shrink_to_fit() to "reverse down" to minimal viable grid (min 16x16 for 4-block)
- Original counting-triggered resize preserved (asymmetric growth for positive)
- Negative values, reverse directions, and negative trends now fully supported without loss
- Demo runs cleanly and plots
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

class SymbolicWave:
    def __init__(self, grid_size: int = 4, start_value: int = 0):
        self.grid_size = grid_size
        self.total_slots = grid_size ** 2
        self.current = start_value
        self.entries: List[str] = []
        self.grid = np.zeros((grid_size * 4, grid_size * 4))
        self.path: List[Tuple[str, int, Tuple[int, int]]] = []  # (sym, value, trend)

        # Corners: (row, col) — row 0 = top, col 0 = left
        self.corners = {
            '[': (0, 0),                               # top-left
            ']': (0, self.grid.shape[1] - 1),          # top-right
            '}': (self.grid.shape[0] - 1, self.grid.shape[1] - 1),  # bottom-right
            '{': (self.grid.shape[0] - 1, 0)            # bottom-left
        }

        # Pi deviation for trend bias
        self.pi_center = np.pi
        self.effective_pi_boundary = 2.078
        self.deviation = self.pi_center - self.effective_pi_boundary  # Asymmetry strength

    def _pad_grid(self, pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0):
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return
        new_height = self.grid.shape[0] + pad_top + pad_bottom
        new_width = self.grid.shape[1] + pad_left + pad_right
        new_grid = np.zeros((new_height, new_width))
        new_grid[pad_top:pad_top + self.grid.shape[0], pad_left:pad_left + self.grid.shape[1]] = self.grid
        self.grid = new_grid
        # Shift corners
        self.corners = {k: (v[0] + pad_top, v[1] + pad_left) for k, v in self.corners.items()}

    def _draw_line(self, start_row: int, start_col: int, end_row: int, end_col: int, value: float):
        """Standard Bresenham line — exact, no clipping needed (bounds ensured by padding)"""
        row = start_row
        col = start_col
        drow = abs(end_row - start_row)
        dcol = abs(end_col - start_col)
        srow = 1 if start_row < end_row else -1
        scol = 1 if start_col < end_col else -1
        err = dcol - drow

        while True:
            self.grid[row, col] += value
            if row == end_row and col == end_col:
                break
            e2 = 2 * err
            if e2 > -drow:
                err -= drow
                col += scol
            if e2 < dcol:
                err += dcol
                row += srow

    def shrink_to_fit(self, min_side: int = 16, margin: int = 20):
        """Reverse down to minimal grid — centers content, clamps corners, updates grid_size"""
        nonzero_rows, nonzero_cols = np.nonzero(self.grid)
        if nonzero_rows.size == 0:
            return

        min_row, max_row = nonzero_rows.min(), nonzero_rows.max()
        min_col, max_col = nonzero_cols.min(), nonzero_cols.max()

        used_height = max_row - min_row + 1
        used_width = max_col - min_col + 1

        new_height = max(min_side, used_height + 2 * margin)
        new_width = max(min_side, used_width + 2 * margin)

        if new_height >= self.grid.shape[0] and new_width >= self.grid.shape[1]:
            return  # No meaningful shrink

        new_grid = np.zeros((new_height, new_width))

        # Center the used content
        offset_row = (new_height - used_height) // 2
        offset_col = (new_width - used_width) // 2

        crop = self.grid[min_row:max_row + 1, min_col:max_col + 1]
        new_grid[offset_row:offset_row + used_height, offset_col:offset_col + used_width] = crop

        self.grid = new_grid

        # Adjust corners relative to new centered position
        adjust_row = min_row - offset_row
        adjust_col = min_col - offset_col
        self.corners = {k: (v[0] - adjust_row, v[1] - adjust_col) for k, v in self.corners.items()}

        # Clamp corners to new bounds
        max_row_new = new_height - 1
        max_col_new = new_width - 1
        self.corners = {k: (max(0, min(v[0], max_row_new)), max(0, min(v[1], max_col_new))) for k, v in self.corners.items()}

        # Update grid_size to match new scale
        side = max(new_height, new_width)
        self.grid_size = max(4, side // 4)
        self.total_slots = self.grid_size ** 2

    def step(self, count: int = 1, direction: int = 1, trend: Tuple[int, int] = (1, 1), multiply: Optional[int] = None, divide: Optional[int] = None):
        for _ in range(count):
            cycle_idx = abs(self.current) % 4
            symbols = ['[', ']', '}', '{'] if direction > 0 else ['{', '}', ']', '[']
            sym = symbols[cycle_idx]

            is_even = self.current % 2 == 0
            if is_even:
                entry = f"({self.current})" if direction > 0 else f"){self.current}("
            else:
                entry = f"{self.current}(" if direction > 0 else f"){self.current}"

            if multiply is not None:
                self.current *= multiply
                entry = f"*{multiply}"
            elif divide is not None:
                self.current //= divide
                entry = f"/{divide}"

            if sym in self.corners:
                start_row, start_col = self.corners[sym]
                amp = 1.0 if is_even else 0.5
                dx, dy = trend  # dx = col direction, dy = row direction
                bias = self.deviation * (dx + dy) / 2

                tentative_end_row = start_row + int(dy * 4 + bias)
                tentative_end_col = start_col + int(dx * 4 + bias)

                min_row = min(start_row, tentative_end_row)
                max_row = max(start_row, tentative_end_row)
                min_col = min(start_col, tentative_end_col)
                max_col = max(start_col, tentative_end_col)

                height, width = self.grid.shape
                margin = 10
                pad_top = max(0, -min_row + margin)
                pad_left = max(0, -min_col + margin)
                pad_bottom = max(0, max_row - height + 1 + margin)
                pad_right = max(0, max_col - width + 1 + margin)

                self._pad_grid(pad_top, pad_bottom, pad_left, pad_right)

                # Updated positions after padding
                start_row += pad_top
                start_col += pad_left
                end_row = tentative_end_row + pad_top
                end_col = tentative_end_col + pad_left

                self._draw_line(start_row, start_col, end_row, end_col, amp)

            self.entries.append(entry)
            self.current += direction

            # Original counting-triggered resize (preserved for asymmetric positive growth)
            if abs(self.current) > self.total_slots * 10:
                self.grid_size *= 2
                self.total_slots = self.grid_size ** 2
                new_grid = np.zeros((self.grid_size * 4, self.grid_size * 4))
                new_grid[:self.grid.shape[0], :self.grid.shape[1]] = self.grid
                self.grid = new_grid
                # Corners remain at old positions (intentional asymmetric embed)

        return self

    def get_string(self) -> str:
        return "".join(self.entries)

    def to_waveform(self) -> np.ndarray:
        signal = self.grid.mean(axis=1)  # Average over columns → vertical "wave"
        signal -= signal.min()
        signal /= signal.max() + 1e-8
        return signal

    def plot_grid_and_wave(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(self.grid, cmap='viridis')
        plt.title("Triangulation Grid")
        plt.axis('off')

        wave = self.to_waveform()
        plt.subplot(1, 3, 2)
        plt.plot(wave)
        plt.title("Reverse Triangulated Waveform")
        plt.ylabel("Amplitude")

        fft = np.fft.rfft(wave)
        freq = np.fft.rfftfreq(len(wave))
        plt.subplot(1, 3, 3)
        plt.plot(freq, np.abs(fft))
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency")

        plt.tight_layout()
        plt.show()

        print("Peak frequency proxy:", np.argmax(np.abs(fft)))

# Demo
if __name__ == "__main__":
    sw = SymbolicWave(grid_size=4)
    sw.step(30, direction=1, trend=(1, 1))    # Diagonal trend
    sw.step(20, direction=1, trend=(-1, 2))   # Different slope (tests negative)
    sw.step(15, direction=-1, trend=(-2, -1)) # Reverse direction + negative trend
    sw.shrink_to_fit()                         # Demonstrate reverse-down to minimal
    print("Final string:", sw.get_string())
    print("Final value:", sw.current)
    print("Final grid shape:", sw.grid.shape)
    sw.plot_grid_and_wave()