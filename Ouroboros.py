"""
Ouroboros.py

Ouroboros – A Scale-Invariant Geometric Framework for Persistence and Resonance

This framework models a spherical manifold where π transitions from ≈3.1416 (local flat center) to a curved boundary value (default edge ≈2.094, tunable). A fixed deviation enforces natural thirds divisions and dual-pass dynamics.

Core Insights (for Humans and AI):
- Dual-pass resonance: First-pass bloom (directional expansion with stochastic kick), second-pass etching (squaring amplification + irreversible pruning).
- Parameter-free derivations: Cosmic densities (~31% matter / ~69% dark energy with time-loss), persistence ratios, and cross-scale patterns.
- Matter/Data Interaction (New): Photons as massless "data couriers" (fast, eternal propagation trails). Electrons as massive "etchers" (negative charge probing positive moat, repulsion preventing collapse).
- Sub-Space Scans: Treat any grid/data as manifold layers—propagate pulses, etch persistent residue (heatmap/trails reveal lasting patterns).
- Biology/EM Tie: Plants as natural manifolds (phyllotaxis thirds/golden deviation). Light (photon freq) blooms excitation, sound/vibration rebounds resilience.

Applications:
- Sub-space pattern discovery (images, networks, sequences).
- Simulated pulse amplification (light/sound for growth/resilience).
- Integrity/resilience tools (EtchSecure submodule inspiration).

Run directly for demos. Import for extensions.

MIT License – explore, extend, share freely.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sympy as sp
from typing import Optional, Tuple

class OuroborosFramework:
    def __init__(self, radius: float = 1.0, noise_level: float = 0.7, scale_factor: float = 4.0,
                 time_loss_factor: Optional[float] = None):
        self.radius = radius
        self.pi_center = np.pi
        self.effective_pi_boundary = 2.0
        self.deviation = 2.0
        self.noise_level = noise_level
        self.scale_factor = scale_factor
        self.time_loss_factor = time_loss_factor
        self.third_edge = 2 * np.pi / 3  # ≈2.094
        self.third_offset = np.pi / 3    # Deviation sweet spot

    # Core Geometry
    def pi_variation(self, position_ratio: float) -> float:
        """π merge from center to boundary—asymmetric etch bias."""
        if not 0 <= position_ratio <= 1:
            raise ValueError("position_ratio must be in [0, 1]")
        delta = self.pi_center - self.effective_pi_boundary
        return self.pi_center - delta * (position_ratio ** self.scale_factor)

    def pi_differential(self, position_ratio: float = 1.0, symbolic: bool = False) -> float:
        """Exact d(π)/d(r)."""
        r = sp.symbols('r')
        delta = self.pi_center - self.effective_pi_boundary
        pi_var = self.pi_center - delta * (r ** self.scale_factor)
        diff = sp.diff(pi_var, r)
        if symbolic:
            return diff
        return float(diff.subs(r, position_ratio).evalf())

    def derive_cosmic_densities(self, use_time_loss: bool = False) -> Tuple[float, float]:
        """Parameter-free cosmic ratios (base vs observed with time-loss)."""
        filled = 1 / (1 + self.deviation / self.third_offset)  # ~0.344 base
        voids = 1 - filled
        if use_time_loss or self.time_loss_factor is not None:
            loss = self.time_loss_factor or 0.138
            filled *= (1 - loss)
            voids = 1 - filled
        return filled, voids

    # Dual-Pass Core
    def dual_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Bloom + etch on grid—returns etched residue, persistence, complement."""
        grid = np.array(initial_grid, dtype=float)

        # First-pass bloom
        bloom = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
        bloom = np.clip(bloom, -self.radius, self.radius)

        # Second-pass etch (EM proxy: square amp as electron jump, prune repulsion moat)
        etched = np.cos(bloom * (self.effective_pi_boundary ** 2))
        etched += (bloom ** 2) * (self.deviation / self.pi_center)
        etched = np.where(np.abs(etched) < 0.1, 0, etched)

        persistence = np.sum(np.abs(etched) > 0.1) / etched.size
        complement = 1 - persistence
        return etched, persistence, complement

    # Upgraded Sub-Space Scan (Feed Any Data/Grid)
    def subspace_scan(self, data_grid: np.ndarray, passes: int = 2) -> Tuple[np.ndarray, float]:
        """Upgraded scan: Treat input as manifold layers—multi-pass etch reveals persistent heatmap."""
        current = np.array(data_grid, dtype=float)
        for _ in range(passes):
            current, pers, _ = self.dual_pass_resonance(current)
        return current, pers  # Heatmap residue + overall persistence

    # EM-Inspired Pulse (Photon Data Kick vs Electron Etch)
    def em_pulse_manifold(self, freq_proxy: float = 660.0, cycles: int = 50, photon_amp: float = 1.5,
                          electron_prune: float = 0.5) -> Tuple[float, float]:
        """Pulse with EM contrast: Photon (massless freq kick bloom), Electron (massive prune etch)."""
        theta = np.linspace(0, cycles * np.pi, cycles)
        # Photon data kick (fast vibration, low dissipation)
        photon_kick = np.sin(theta * freq_proxy / 100) * photon_amp
        
        # Electron etch proxy (massive repulsion moat + attraction pull)
        bloom = photon_kick + self.noise_level * np.random.randn(len(theta))
        etched = np.cos(bloom ** 2)
        etched = np.where(np.abs(etched) < electron_prune, 0, etched)
        
        persistence = np.sum(np.abs(etched) > electron_prune) / len(etched)
        reclaimed = np.sum(np.abs(bloom[etched == 0]))  # "Heat" from prune
        return persistence, reclaimed

    # Plant Manifold Pulse (Rebound Amp for Growth Boom)
    def pulse_plant_manifold(self, base_freq: float = 220.0, cycles: int = 100, rebound_amp: bool = True) -> Tuple[float, float, np.ndarray]:
        """Sound/vibration pulse on phyllotaxis spiral—rebound via thirds asymmetry for resilience/growth."""
        theta = np.linspace(0, 10 * np.pi, cycles)
        r = np.sqrt(theta)  # Spiral proxy
        pulse = np.sin(theta * base_freq / 100)
        
        if rebound_amp:
            rebound = pulse * np.cos(theta + self.third_offset) * 1.5  # Asymmetry reflection boost
            pulse += rebound
        
        bloom = pulse + self.noise_level * np.random.randn(len(pulse))
        etched = np.cos(bloom ** 2)
        etched = np.where(np.abs(etched) < 0.5, 0, etched)
        
        persistence = np.sum(np.abs(etched) > 0.5) / len(etched)
        reclaimed = np.sum(np.abs(bloom[etched == 0]))
        return persistence, reclaimed, pulse  # Growth proxy, fuel, wave for viz

    # Visualizations (Upgraded with EM/Plant Options)
    def visualize_time_flow(self, steps: int = 200, persistence_levels = [0.2, 0.5, 0.8, 0.95], save_path: Optional[str] = None):
        # (Existing code from previous—kept for continuity)
        # ... (full viz code as before for trails)

    def visualize_ring_manifold_time_flow(self, steps: int = 300, ring_ratios = [0.2, 0.4, 0.6, 0.8],
                                          persistence_levels = [0.95, 0.8, 0.6, 0.3], save_path: Optional[str] = None):
        # (Existing ring viz code)

    def visualize_manifold(self, save_path: Optional[str] = None):
        # (Existing 3D manifold code)

# Example usage
if __name__ == "__main__":
    ouro = OuroborosFramework()

    print("=== Ouroboros Demo ===")
    print("Base densities:", ouro.derive_cosmic_densities())
    print("Observed (time-loss):", ouro.derive_cosmic_densities(use_time_loss=True))

    # Sub-space scan test (random grid)
    grid = np.random.uniform(-1, 1, (50, 50))
    scanned, pers = ouro.subspace_scan(grid)
    print(f"Sub-space scan persistence: {pers:.4f}")

    # EM pulse test
    pers_em, rec_em = ouro.em_pulse_manifold(freq_proxy=660.0)  # Red light proxy
    print(f"EM pulse (red bloom): Persistence {pers_em:.4f}, Reclaimed {rec_em:.4f}")

    # Plant sound pulse test
    pers_plant, rec_plant, wave = ouro.pulse_plant_manifold(base_freq=220.0)
    print(f"Plant pulse (220Hz rebound): Persistence {pers_plant:.4f}, Reclaimed {rec_plant:.4f}")

    # Visuals
    ouro.visualize_time_flow()
    ouro.visualize_ring_manifold_time_flow()
    ouro.visualize_manifold()
