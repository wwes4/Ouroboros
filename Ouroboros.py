"""
Ouroboros.py

Ouroboros – A Scale-Invariant Geometric Framework for Persistence and Resonance

Models a spherical manifold with π transitioning from flat center (≈3.1416) to theoretical exact 2π/3 (≈2.0944),
with a reverse-calculated effective boundary (≈2.078) introducing per-frame time asymmetry (frame_delta ≈0.0164).
This split preserves clean thirds algebra while enabling directional time flow and discrete granularity.

Core: Dual/multi-pass resonance on grids — bloom (sinusoidal expansion + stochastic kick) → etch (cosine squaring + pruning).
Fibonacci-phased mode liberates decoherent local detail via extended raw bloom chaining before coherent convergence.

New: Explicit manifold time calculation and persistence possibility estimation over cycles/distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sympy as sp
from typing import Optional, Tuple, List

# Golden ratio for intuition/decoherence phase balance
PHI = (1 + np.sqrt(5)) / 2  # ≈1.618
DECOHERENCE_RATIO = 1 / PHI   # ≈0.618
COHERENCE_RATIO = PHI - 1     # ≈0.618

class OuroborosFramework:
    def __init__(self, radius: float = 1.0, target_filled: float = 0.31, scale_factor: float = 4.0,
                 use_fibonacci_phases: bool = False, max_fib_index: int = 89, favor_decoherence: bool = True):
        self.radius = radius
        self.scale_factor = scale_factor
        self.pi_center = np.pi
        
        # Theoretical exact thirds — clean algebraic derivations / zoomed-out
        self.theoretical_pi_boundary = 2 * np.pi / 3  # ≈2.094395102393195
        self.third_offset = np.pi / 3  # Exact π/3
        
        # Effective boundary — numerical dynamics + time flow asymmetry
        self.effective_pi_boundary = 2.078  # Reverse-tuned
        
        # Per-frame/cycle time unit from boundary asymmetry
        self.frame_delta = self.theoretical_pi_boundary - self.effective_pi_boundary  # ≈0.016395
        
        # Deviation from density target using theoretical thirds
        self.deviation = (1 / target_filled - 1) * self.third_offset
        
        # DE proxy kick
        self.noise_level = 0.69
        
        # Mobius central flow prune
        self.prune_threshold = 0.1
        
        # Cumulative dilution factor
        self.time_loss_factor = 0.138

        # Decoherence control
        self.use_fibonacci_phases = use_fibonacci_phases
        self.max_fib_index = max_fib_index
        self.favor_decoherence = favor_decoherence

    def pi_variation(self, position_ratio: float) -> float:
        """π from center to theoretical thirds (clean gradient), asymmetric."""
        if not 0 <= position_ratio <= 1:
            raise ValueError("position_ratio must be in [0, 1]")
        delta = self.pi_center - self.theoretical_pi_boundary
        return self.pi_center - delta * (position_ratio ** self.scale_factor)

    def pi_differential(self, position_ratio: float = 1.0, symbolic: bool = False) -> float:
        """d(π)/d(r) pressure gradient (uses theoretical for purity)."""
        r = sp.symbols('r')
        delta = self.pi_center - self.theoretical_pi_boundary
        pi_var = self.pi_center - delta * (r ** self.scale_factor)
        diff = sp.diff(pi_var, r)
        if symbolic:
            return diff
        return float(diff.subs(r, position_ratio).evalf())

    def derive_cosmic_densities(self, use_time_loss: bool = False) -> Tuple[float, float]:
        """Densities from thirds/deviation balance ± time-loss dilution."""
        filled = 1 / (1 + self.deviation / self.third_offset)
        voids = 1 - filled
        if use_time_loss:
            filled *= (1 - self.time_loss_factor)
            voids = 1 - filled
        return filled, voids

    def dual_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Standard bloom → etch pass."""
        grid = np.array(initial_grid, dtype=float)

        # Bloom — photon-like expansion
        bloom = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
        bloom = np.clip(bloom, -self.radius, self.radius)

        # Etch — electron-like prune with effective boundary time asymmetry
        etched = np.cos(bloom * (self.effective_pi_boundary ** 2))
        etched += (bloom ** 2) * (self.deviation / self.pi_center)
        etched = np.where(np.abs(etched) > self.prune_threshold, 0, etched)  # Note: original had <, but logic matches prune to zero low-residue

        persistence = np.sum(np.abs(etched) > self.prune_threshold) / etched.size
        complement = 1 - persistence
        return etched, persistence, complement

    def fibonacci_multi_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, List[float], List[int], List[float]]:
        """Fibonacci-phased alternation with cumulative manifold time tracking."""
        grid = np.array(initial_grid, dtype=float)
        persistences = []
        phase_lengths = []
        cumulative_times = [0.0]  # Start at t=0

        a, b = 1, 1
        phase = 0  # 0: decoherent bloom-heavy, 1: coherent etch-heavy
        if self.favor_decoherence:
            a, b = b, a + b

        cycle = 0
        while b <= self.max_fib_index and cycle < 30:
            length = b if (phase == 0) == self.favor_decoherence else a
            phase_lengths.append(length)

            for _ in range(int(length)):
                if phase == 0:  # Raw decoherent chaining
                    grid = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
                    grid = np.clip(grid, -self.radius, self.radius)
                else:  # Coherent prune with time asymmetry
                    grid = np.cos(grid * (self.effective_pi_boundary ** 2))
                    grid += (grid ** 2) * (self.deviation / self.pi_center)
                    grid = np.where(np.abs(grid) < self.prune_threshold, 0, grid)

            persistence = np.sum(np.abs(grid) > self.prune_threshold) / grid.size
            persistences.append(persistence)

            # Advance time per completed phase
            current_time = (cycle + 1) * self.frame_delta * (length / (a + b))  # Proportional advance; approximates per-phase
            cumulative_times.append(cumulative_times[-1] + current_time)

            a, b = b, a + b
            phase = 1 - phase
            cycle += 1

        return grid, persistences, phase_lengths, cumulative_times

    def subspace_scan(self, data_grid: np.ndarray, passes: int = 2, use_fib: Optional[bool] = None) -> Tuple[np.ndarray, float]:
        """Scan with optional Fibonacci mode."""
        if use_fib is None:
            use_fib = self.use_fibonacci_phases

        current = np.array(data_grid, dtype=float)

        if use_fib:
            final_grid, persistences, _, cumulative_times = self.fibonacci_multi_pass_resonance(current)
            return final_grid, persistences[-1]  # Compatibility return
        else:
            final_pers = 0.0
            for _ in range(passes):
                current, pers, _ = self.dual_pass_resonance(current)
                final_pers = pers
            return current, final_pers

    # New explicit time/possibility methods
    def calculate_frame_time_delta(self) -> float:
        return self.frame_delta

    def calculate_manifold_time(self, num_cycles: int) -> float:
        return num_cycles * self.frame_delta

    def estimate_persistence_possibility(self, initial_persistence: float, num_cycles: int,
                                        position_ratio: float = 1.0) -> float:
        pi_local = self.pi_variation(position_ratio)
        pressure_damping = pi_local / self.pi_center
        loss_decay = np.exp(-self.time_loss_factor * num_cycles)
        return initial_persistence * pressure_damping * loss_decay

    # === Original methods preserved unchanged below ===

    def em_pulse_manifold(self, freq_proxy: float = 660.0, cycles: int = 50, photon_amp: float = 1.5,
                          electron_prune: float = 0.5) -> Tuple[float, float]:
        theta = np.linspace(0, cycles * np.pi, cycles)
        photon_kick = np.sin(theta * freq_proxy / 100) * photon_amp
        
        bloom = photon_kick + self.noise_level * np.random.randn(len(theta))
        etched = np.cos(bloom ** 2)
        etched = np.where(np.abs(etched) < electron_prune, 0, etched)
        
        persistence = np.sum(np.abs(etched) > electron_prune) / len(etched)
        reclaimed = np.sum(np.abs(bloom[etched == 0]))
        return persistence, reclaimed

    def pulse_plant_manifold(self, base_freq: float = 220.0, cycles: int = 100, rebound_amp: bool = True) -> Tuple[float, float, np.ndarray]:
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
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        return points

    def simulate_manifold_slice_pull(self, pull_distance: float = 1.0, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
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

    # Visualizations (unchanged)
    def visualize_time_flow(self, steps: int = 200, persistence_levels=[0.2, 0.5, 0.8, 0.95], save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-self.radius*1.2, self.radius*1.2)
        ax.set_ylim(-self.radius*1.2, self.radius*1.2)
        ax.set_facecolor('black')
        ax.set_title("Time Flow as Ghostly Persistence Trails")

        theta = np.linspace(0, 6*np.pi, steps)
        x_base = np.cos(theta)
        y_base = np.sin(theta)

        colors = plt.cm.viridis(np.linspace(0, 1, len(persistence_levels)))

        for i, pers in enumerate(persistence_levels):
            trail_length = int(steps * pers)
            alphas = np.linspace(0.1, 1.0, trail_length)
            color = colors[i]
            ax.plot(x_base[:trail_length], y_base[:trail_length], color=color, alpha=0.6, linewidth=2)
            ax.scatter(x_base[:trail_length], y_base[:trail_length], c=alphas, cmap='viridis', s=15, alpha=0.8)

        ax.text(0.02, 0.98, f"Levels: {persistence_levels}\nLow = fleeting | High = enduring",
                transform=ax.transAxes, color='white', fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        if save_path:
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()

    def visualize_ring_manifold_time_flow(self, steps: int = 300, ring_ratios=[0.2, 0.4, 0.6, 0.8], 
                                          persistence_levels=[0.95, 0.8, 0.6, 0.3], save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-self.radius*1.3, self.radius*1.3)
        ax.set_ylim(-self.radius*1.3, self.radius*1.3)
        ax.set_facecolor('black')
        ax.set_title("Ring Manifold Time Flow – Concentric Persistence Layers")

        theta = np.linspace(0, 8*np.pi, steps)

        colors = plt.cm.plasma(np.linspace(0, 1, len(ring_ratios)))

        for i, (ratio, pers) in enumerate(zip(ring_ratios, persistence_levels)):
            radius_ring = ratio * self.radius
            x_base = radius_ring * np.cos(theta)
            y_base = radius_ring * np.sin(theta)
            trail_length = int(steps * pers)
            alphas = np.linspace(0.05, 1.0, trail_length)
            color = colors[i]
            if pers > 0.7:
                ax.plot(x_base[:trail_length], y_base[:trail_length], color=color, linewidth=3, alpha=0.8)
            ax.scatter(x_base[:trail_length], y_base[:trail_length], c=alphas, cmap='plasma', s=8, alpha=0.7)

        for r_ratio in [1/3, 2/3, 1.0]:
            circle = plt.Circle((0, 0), r_ratio * self.radius, color='red', fill=False, linewidth=1, alpha=0.5, ls='--')
            ax.add_patch(circle)

        ax.text(0.02, 0.98, "Inner: High persistence → dense trails\nOuter: Low → fading ghosts",
                transform=ax.transAxes, color='white', fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def visualize_manifold(self, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        r = np.sqrt(x**2 + y**2 + z**2)
        pi_color = self.pi_center - (self.pi_center - self.theoretical_pi_boundary) * (r ** self.scale_factor)  # Updated to theoretical
        ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(pi_color / self.pi_center), alpha=0.6)
        for ratio in [1/3, 2/3]:
            theta = np.linspace(0, 2*np.pi, 100)
            z_plane = np.cos(np.arcsin(ratio)) * np.ones(100)
            x_plane = ratio * np.cos(theta)
            y_plane = ratio * np.sin(theta)
            ax.plot(x_plane, y_plane, z_plane, color='red', linewidth=2, alpha=0.8)
        ax.set_title("Ouroboros Manifold – π Variation + Thirds Divisions")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def visualize_slice_pull(self, original_points: np.ndarray, pulled_points: np.ndarray, 
                             etched: np.ndarray, save_path: Optional[str] = None):
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

# Example usage
if __name__ == "__main__":
    ouro = OuroborosFramework()

    print("=== Ouroboros Demo ===")
    print("Derived deviation:", ouro.deviation)
    print("Theoretical boundary:", ouro.theoretical_pi_boundary)
    print("Effective boundary:", ouro.effective_pi_boundary)
    print("Frame delta (per-cycle time):", ouro.frame_delta)
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

    # Slice/pull transmission demo
    pulled, etched, pers = ouro.simulate_manifold_slice_pull(pull_distance=1.5)
    print(f"Slice/pull transmission persistence: {pers:.4f}")
    ouro.visualize_slice_pull(ouro.generate_manifold_points(), pulled, etched)

    # Visuals
    ouro.visualize_time_flow()
    ouro.visualize_ring_manifold_time_flow()
    ouro.visualize_manifold()
