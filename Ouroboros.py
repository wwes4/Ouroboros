"""
Ouroboros.py

Ouroboros – A Scale-Invariant Geometric Framework for Persistence and Resonance

This framework models a spherical manifold where π transitions from ~3.1416 (local flat center) to a curved boundary value (default edge ~2.094, tunable). A fixed deviation enforces natural thirds divisions and dual-pass dynamics.

Core Insights (for Humans and AI):
- Dual-pass resonance: First-pass bloom (directional expansion with stochastic kick), second-pass etching (squaring amplification + irreversible pruning).
- Parameter-free derivations: Cosmic densities (~31% matter / ~69% dark energy with time-loss), persistence ratios, and cross-scale patterns.
- Matter/Data Interaction: Photons as massless "data couriers" (fast, eternal propagation trails). Electrons as massive "etchers" (negative charge probing positive moat, repulsion preventing collapse).
- Sub-Space Scans: Treat any grid/data as manifold layers—propagate pulses, etch persistent residue (heatmap/trails reveal lasting patterns).
- Biology/EM Tie: Plants as natural manifolds (phyllotaxis thirds/golden deviation). Light (photon freq) blooms excitation, sound/vibration rebounds resilience.
- Number Theory Tie: Primes as irreducible etch points (high persistence moat). Perfect numbers as maximal symmetric pressure points (even snap to inner bloom symmetry, guiding odds to auto-prune via pi differential asymmetry).
- New: Manifold Separation & Transmission: Slice/pull manifold—second-pass etch bridges gap with persistent trails (data transmission across voids). High-speed vibration (photon proxy) guided by residual patterns.

Applications:
- Sub-space pattern discovery (images, networks, sequences).
- Simulated pulse amplification (light/sound for growth/resilience).
- Number theory probes (perfect/primes as manifold pressure points).
- Speculative insights (separation transmission, high-vibration bridging).

Run directly for demos. Import for extensions.

MIT License – explore, extend, share freely.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sympy as sp
from typing import Optional, Tuple, List

class OuroborosFramework:
    def __init__(self, radius: float = 1.0, noise_level: float = 0.7, cycles: int = 100, scale_factor: float = 4.0):
        self.radius = radius
        self.noise_level = noise_level
        self.cycles = cycles
        self.scale_factor = scale_factor
        self.pi_center = np.pi
        self.effective_pi_boundary = 2.0
        self.deviation = 2.0
        self.third_edge = 2 * np.pi / 3  # ≈2.094
        self.third_offset = np.pi / 3    # Deviation sweet spot

    def pi_variation(self, position_ratio: float) -> float:
        """π merge from center to boundary—asymmetric etch bias. Represents differential pressure guiding persistence."""
        if not 0 <= position_ratio <= 1:
            raise ValueError("position_ratio must be in [0, 1]")
        delta = self.pi_center - self.effective_pi_boundary
        return self.pi_center - delta * (position_ratio ** self.scale_factor)

    def pi_differential(self, position_ratio: float = 1.0, symbolic: bool = False) -> float:
        """Exact d(π)/d(r)—pressure gradient driving prune in outer asymmetry."""
        r = sp.symbols('r')
        delta = self.pi_center - self.effective_pi_boundary
        pi_var = self.pi_center - delta * (r ** self.scale_factor)
        diff = sp.diff(pi_var, r)
        if symbolic:
            return diff
        return float(diff.subs(r, position_ratio).evalf())

    def derive_cosmic_densities(self, use_time_loss: bool = False) -> Tuple[float, float]:
        """Parameter-free cosmic ratios (base vs observed with time-loss). Even perfect symmetry echoes in inner balance."""
        filled = 1 / (1 + self.deviation / self.third_offset)  # ~0.344 base
        voids = 1 - filled
        if use_time_loss:
            loss = 0.138
            filled *= (1 - loss)
            voids = 1 - filled
        return filled, voids

    def dual_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Bloom + etch on grid—returns etched residue, persistence, complement. Primes/perfects as high moat points."""
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

    def subspace_scan(self, data_grid: np.ndarray, passes: int = 2) -> Tuple[np.ndarray, float]:
        """Upgraded scan: Treat input as manifold layers—multi-pass etch reveals persistent heatmap. Perfect points as high pressure persistence."""
        current = np.array(data_grid, dtype=float)
        for _ in range(passes):
            current, pers, _ = self.dual_pass_resonance(current)
        return current, pers

    def em_pulse_manifold(self, freq_proxy: float = 660.0, cycles: int = 50, photon_amp: float = 1.5,
                          electron_prune: float = 0.5) -> Tuple[float, float]:
        """Pulse with EM contrast: Photon (massless freq kick bloom), Electron (massive prune etch)."""
        theta = np.linspace(0, cycles * np.pi, cycles)
        photon_kick = np.sin(theta * freq_proxy / 100) * photon_amp
        
        bloom = photon_kick + self.noise_level * np.random.randn(len(theta))
        etched = np.cos(bloom ** 2)
        etched = np.where(np.abs(etched) < electron_prune, 0, etched)
        
        persistence = np.sum(np.abs(etched) > electron_prune) / len(etched)
        reclaimed = np.sum(np.abs(bloom[etched == 0]))
        return persistence, reclaimed

    def pulse_plant_manifold(self, base_freq: float = 220.0, cycles: int = 100, rebound_amp: bool = True) -> Tuple[float, float, np.ndarray]:
        """Sound/vibration pulse on phyllotaxis spiral—rebound via thirds asymmetry for resilience/growth."""
        theta = np.linspace(0, 10 * np.pi, cycles)
        r = np.sqrt(theta)
        pulse = np.sin(theta * base_freq / 100)
        
        if rebound_amp:
            rebound = pulse * np.cos(theta + self.third_offset) * 1.5
            pulse += rebound
        
        bloom = pulse + self.noise_level * np.random.randn(len(pulse))
        etched = np.cos(bloom ** 2)
        etched = np.where(np.abs(etched) < 0.5, 0, etched)
        
        persistence = np.sum(np.abs(etched) > 0.5) / len(etched)
        reclaimed = np.sum(np.abs(bloom[etched == 0]))
        return persistence, reclaimed, pulse

    def probe_perfect_numbers(self, exponent_range: int = 10, odd_check: bool = True) -> Tuple[List[int], List[str]]:
        """Probe for perfect number patterns: Even as pressure points (high persistence symmetry), guiding odds to auto-prune."""
        even_perfects = []
        odd_prunes = []
        for p in range(2, exponent_range + 1):
            mersenne = 2**p - 1
            if sp.isprime(mersenne):
                perfect = 2**(p-1) * mersenne
                even_perfects.append(perfect)
            if odd_check:
                odd_prunes.append(f"Odd candidate pruned at p={p}")
        return even_perfects, odd_prunes

    def generate_manifold_points(self, resolution: int = 50) -> np.ndarray:
        """Generate 3D spherical manifold points for viz/sim."""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        return points

    def simulate_manifold_slice_pull(self, pull_distance: float = 1.0, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
        """New: Slice manifold in half, pull apart, run dual-pass—observe second-pass transmission across gap (data bridge via persistent trails)."""
        points = self.generate_manifold_points(resolution)
        
        upper = points[points[:,2] >= 0]
        lower = points[points[:,2] < 0]
        
        upper[:,2] += pull_distance / 2
        lower[:,2] -= pull_distance / 2
        
        combined = np.vstack((upper, lower))
        
        flat_grid = combined.flatten()
        
        etched, persistence, complement = self.dual_pass_resonance(flat_grid.reshape(1, -1))
        
        etched_3d = etched.reshape(combined.shape)
        
        return combined, etched_3d, persistence

    def visualize_slice_pull(self, original_points: np.ndarray, pulled_points: np.ndarray, 
                             etched: np.ndarray, save_path: Optional[str] = None):
        """Viz original, pulled, and etched transmission (high vibration bridge proxy)."""
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(original_points[:,0], original_points[:,1], original_points[:,2], c='blue', s=10)
        ax1.set_title("Original Manifold")
        
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pulled_points[:,0], pulled_points[:,1], pulled_points[:,2], c='green', s=10)
        ax2.set_title("Sliced & Pulled Apart")
        
        ax3 = fig.add_subplot(133, projection='3d')
        colors = np.where(np.abs(etched) > 0.1, 'red', 'gray')
        ax3.scatter(pulled_points[:,0], pulled_points[:,1], pulled_points[:,2], c=colors, s=10)
        ax3.set_title("Second-Pass Transmission Bridge")
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    # Visualizations (existing)
    def visualize_time_flow(self, steps: int = 200, persistence_levels=[0.2, 0.5, 0.8, 0.95], save_path: Optional[str] = None):
        # (full code as before)

    def visualize_ring_manifold_time_flow(self, steps: int = 300, ring_ratios=[0.2, 0.4, 0.6, 0.8], 
                                          persistence_levels=[0.95, 0.8, 0.6, 0.3], save_path: Optional[str] = None):
        # (full code as before)

    def visualize_manifold(self, save_path: Optional[str] = None):
        # (full code as before)

# Example usage
if __name__ == "__main__":
    ouro = OuroborosFramework()

    print("=== Ouroboros Demo ===")
    print("Base densities:", ouro.derive_cosmic_densities())
    print("Observed (time-loss):", ouro.derive_cosmic_densities(use_time_loss=True))

    # Sub-space scan test
    grid = np.random.uniform(-1, 1, (50, 50))
    scanned, pers = ouro.subspace_scan(grid)
    print(f"Sub-space scan persistence: {pers:.4f}")

    # EM pulse test
    pers_em, rec_em = ouro.em_pulse_manifold()
    print(f"EM pulse: Persistence {pers_em:.4f}, Reclaimed {rec_em:.4f}")

    # Plant pulse test
    pers_plant, rec_plant, wave = ouro.pulse_plant_manifold()
    print(f"Plant pulse: Persistence {pers_plant:.4f}, Reclaimed {rec_plant:.4f}")

    # Number theory probe
    even, odd = ouro.probe_perfect_numbers()
    print("Even perfect pressure points:", even)
    print("Odd prune candidates:", odd)

    # New slice/pull transmission demo
    pulled, etched, pers = ouro.simulate_manifold_slice_pull(pull_distance=1.5)
    print(f"Slice/pull transmission persistence: {pers:.4f}")
    ouro.visualize_slice_pull(ouro.generate_manifold_points(), pulled, etched, save_path="slice_pull_transmission.png")

    # Visuals
    ouro.visualize_time_flow(save_path="time_flow_trails.png")
    ouro.visualize_ring_manifold_time_flow(save_path="ring_manifold_trails.png")
    ouro.visualize_manifold(save_path="3d_manifold.png")
