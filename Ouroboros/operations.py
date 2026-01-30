"""
operations.py — Low-level persistence operations + vibration extensions
Core bloom/etch/prune from slim Ouroboros, with optional vibration integrations
for feedback loops (e.g., in tri-pass or agentic runs).
"""

import numpy as np
from typing import List, Dict, Optional

# Vibration extensions (imported from vibration.py)
from vibration import propagate_vibration, refract, holographic_linkage

def bloom(grid: np.ndarray, noise_amp: float, pi_center: float) -> np.ndarray:
    expanded = np.sin(grid * pi_center) * (1 + noise_amp)
    return expanded + np.random.uniform(-noise_amp, noise_amp, grid.shape)

def etch(grid: np.ndarray, effective_pi_boundary: float, deviation: float, pi_center: float) -> np.ndarray:
    constrained = np.cos(grid) * (grid ** 2 < effective_pi_boundary ** 2)
    return constrained + deviation * np.tanh(grid / pi_center)

def prune(grid: np.ndarray, prune_threshold: float) -> np.ndarray:
    grid[np.abs(grid) < prune_threshold] = 0
    return grid

def apply_library_feedback(
    grid: np.ndarray,
    truth_library: List[Dict],
    env_feedback_fraction: float,
    framework: Optional[object] = None  # Optional framework reference for vibration extensions
) -> np.ndarray:
    """
    Applies truth library feedback with optional vibration/resonance enhancements.
    If framework is provided, can integrate holographic_linkage or propagation for richer etching.
    """
    if not truth_library:
        return grid
    
    feedback = np.zeros_like(grid)
    target_len = grid.size
    
    for item in truth_library:
        proj = np.array(item["projected"])
        resampled = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(proj)), proj)
        feedback += resampled.reshape(grid.shape) * env_feedback_fraction
    
    # Optional vibration enhancement: holographic linkage if framework available
    if framework is not None:
        feedback = holographic_linkage(framework, feedback)
    
    return grid + feedback

# Example usage in tri-pass (can call from OuroborosFramework):
# feedback = apply_library_feedback(grid, self.truth_library, self.env_feedback_fraction, framework=self)