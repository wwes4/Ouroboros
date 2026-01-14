"""
SymbolicWave.py — Triangulation & Waveform Encoder for Vast Numbers
January 14, 2026

Implements the exact symbolic counting system with:
- 4-corner triangulation ([ top-left, ] top-right, } bottom-right, { bottom-left)
- Consecutive ±1 movement with ( ) for odd/even placement
- * multiply (time-biased: *1 = +1 flow), / divide
- 0 data transmission placeholder
- Dynamic scaling exponents and grid auto-adjustment
- Trend vectors (pi deviation bias) for directed asymmetry
- Building (full values) vs compact (closure/stop signal)
- String compilation and reverse triangulation
- Vibrational waveform calculation via reverse triangulation (FFT proxy + plot)
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
        self.path: List[Tuple[str, int, Tuple[int, int]]] = []

        self.corners = {
            '[': (0, 0),
            ']': (0, self.grid_size*4 - 1),
            '}': (self.grid_size*4 - 1, self.grid_size*4 - 1),
            '{': (self.grid_size*4 - 1, 0)
        }

        self.pi_center = np.pi
        self.effective_pi_boundary = 2.078
        self.deviation = self.pi_center - self.effective_pi_boundary

    def _draw_line(self, start: Tuple[int, int], end: Tuple[int, int], value: float):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.grid[y0, x0] += value
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def step(self, count: int = 1, direction: int = 1, trend: Tuple[int, int] = (1, 1), multiply: Optional[int] = None, divide: Optional[int] = None):
        """Building step — full values for ongoing flow."""
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
                if multiply == 1:
                    self.current += direction  # Time-biased *1 = +1 flow
                else:
                    self.current *= multiply
                entry = f"*{multiply}"
            elif divide is not None:
                self.current //= divide
                entry = f"/{divide}"

            if sym in self.corners:
                corner = self.corners[sym]
                amp = 1.0 if is_even else 0.5
                dx, dy = trend
                bias = self.deviation * (dx + dy) / 2
                end = (corner[0] + dx * 4 + bias, corner[1] + dy * 4 + bias)
                self._draw_line(corner, end, amp)

            self.entries.append(entry)
            self.current += direction

            if abs(self.current) > self.total_slots * 10:
                self.grid_size *= 2
                self.total_slots = self.grid_size ** 2
                new_grid = np.zeros((self.grid_size * 4, self.grid_size * 4))
                new_grid[:self.grid.shape[0], :self.grid.shape[1]] = self.grid
                self.grid = new_grid

        return self

    def compact_closure(self):
        """Compact mirror for box end/stop signal."""
        if self.entries:
            last = self.entries[-1]
            # Simple compact: abbreviate last matching pair
            self.entries[-1] = last[:3] + "}" if last.endswith(")") else last + "}"

    def get_string(self) -> str:
        return "".join(self.entries)

    def to_waveform(self) -> np.ndarray:
        signal = self.grid.mean(axis=1)
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
    sw.step(30, direction=1, trend=(1, 1))
    sw.compact_closure()  # Signal stop
    print("Final string with compact closure:", sw.get_string())
    sw.plot_grid_and_wave()
