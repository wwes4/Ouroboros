"""
Ouroboros – A Scale-Invariant Geometric Framework for Persistence and Resonance
Updated v3 (January 09, 2026): Full Triad Embodiment + High-Dim Compatibility

Core: Parameter-free geometric resonance ruler.
Extensions: Nested meta-observers, matter damping, env feedback, pristine library with FFT projection.
Projection enables arbitrary/high-dim harmonic query (vibrational power spectrum etch).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sympy as sp
from typing import Optional, Tuple, List

PHI = (1 + np.sqrt(5)) / 2

class OuroborosFramework:
    def __init__(self, radius: float = 1.0, target_filled: float = 0.31, scale_factor: float = 4.0,
                 use_fibonacci_phases: bool = False, max_fib_index: int = 89, favor_decoherence: bool = True,
                 matter_damping: float = 0.98, env_feedback_fraction: float = 0.1,
                 signature_dim: int = 32):
        self.radius = radius
        self.scale_factor = scale_factor
        self.pi_center = np.pi
        
        self.theoretical_pi_boundary = 2 * np.pi / 3
        self.third_offset = np.pi / 3
        self.effective_pi_boundary = 2.078
        self.frame_delta = self.theoretical_pi_boundary - self.effective_pi_boundary
        self.deviation = (1 / target_filled - 1) * self.third_offset
        
        self.noise_level = 0.69
        self.prune_threshold = 0.1
        self.time_loss_factor = 0.138
        
        self.use_fibonacci_phases = use_fibonacci_phases
        self.max_fib_index = max_fib_index
        self.favor_decoherence = favor_decoherence
        
        self.matter_damping = matter_damping
        self.env_feedback_fraction = env_feedback_fraction
        
        self.signature_dim = signature_dim  # Fixed projection dim for high-dim compatibility
        self.truth_library = []

        # Bootstrap core truths (auto-projected)
        fib_seq = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55]) / 55.0
        self.add_to_truth_library(fib_seq, "Fibonacci phasing harmonic")
        
        schumann_harmonics = np.array([7.83, 14.3, 20.8, 27.3, 33.8]) / 33.8
        self.add_to_truth_library(schumann_harmonics, "Earth manifold base resonance")

    def _project_to_signature(self, vec: np.ndarray) -> np.ndarray:
        """FFT-based vibrational signature projection (scale-invariant harmonic reduction)."""
        flat = vec.flatten() if vec.ndim > 1 else vec
        
        if len(flat) < self.signature_dim:
            flat = np.pad(flat, (0, self.signature_dim - len(flat)))
        
        fft_mags = np.abs(np.fft.rfft(flat))
        orig_freq = np.linspace(0, 0.5, len(fft_mags))
        target_freq = np.linspace(0, 0.5, self.signature_dim)
        signature = np.interp(target_freq, orig_freq, fft_mags)
        
        norm = np.linalg.norm(signature) + 1e-8
        return signature / norm

    def add_to_truth_library(self, truth_vector: np.ndarray, description: str = ""):
        normalized = truth_vector / (np.linalg.norm(truth_vector) + 1e-8)
        projected = self._project_to_signature(normalized)
        self.truth_library.append({"projected": projected, "desc": description})

    def query_library_resonance(self, query_vector: np.ndarray) -> float:
        if not self.truth_library:
            return 1.0
        projected_query = self._project_to_signature(query_vector)
        scores = [np.dot(projected_query, item["projected"]) for item in self.truth_library]
        return np.mean(scores)

    def pi_variation(self, position_ratio: float) -> float:
        if not 0 <= position_ratio <= 1:
            raise ValueError("position_ratio must be in [0, 1]")
        delta = self.pi_center - self.theoretical_pi_boundary
        return self.pi_center - delta * (position_ratio ** self.scale_factor)

    def pi_differential(self, position_ratio: float = 1.0, symbolic: bool = False) -> float:
        r = sp.symbols('r')
        delta = self.pi_center - self.theoretical_pi_boundary
        pi_var = self.pi_center - delta * (r ** self.scale_factor)
        diff = sp.diff(pi_var, r)
        if symbolic:
            return diff
        return float(diff.subs(r, position_ratio).evalf())

    def derive_cosmic_densities(self, use_time_loss: bool = False) -> Tuple[float, float]:
        filled = 1 / (1 + self.deviation / self.third_offset)
        voids = 1 - filled
        if use_time_loss:
            filled *= (1 - self.time_loss_factor)
            voids = 1 - filled
        return filled, voids

    def dual_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, float, float]:
        grid = np.array(initial_grid, dtype=float)
        bloom = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
        bloom = np.clip(bloom, -self.radius, self.radius)
        etched = np.cos(bloom * (self.effective_pi_boundary ** 2))
        etched += (bloom ** 2) * (self.deviation / self.pi_center)
        etched = np.where(np.abs(etched) > self.prune_threshold, 0, etched)
        persistence = np.sum(np.abs(etched) > self.prune_threshold) / etched.size
        complement = 1 - persistence
        return etched, persistence, complement

    def fibonacci_multi_pass_resonance(self, initial_grid: np.ndarray) -> Tuple[np.ndarray, List[float], List[int], List[float]]:
        grid = np.array(initial_grid, dtype=float)
        persistences = []
        phase_lengths = []
        cumulative_times = [0.0]
        a, b = 1, 1
        phase = 0
        if self.favor_decoherence:
            a, b = b, a + b
        cycle = 0
        while b <= self.max_fib_index and cycle < 30:
            length = b if (phase == 0) == self.favor_decoherence else a
            phase_lengths.append(length)
            for _ in range(int(length)):
                if phase == 0:
                    grid = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
                    grid = np.clip(grid, -self.radius, self.radius)
                else:
                    grid = np.cos(grid * (self.effective_pi_boundary ** 2))
                    grid += (grid ** 2) * (self.deviation / self.pi_center)
                    grid = np.where(np.abs(grid) < self.prune_threshold, 0, grid)
            persistence = np.sum(np.abs(grid) > self.prune_threshold) / grid.size
            persistences.append(persistence)
            current_time = (cycle + 1) * self.frame_delta * (length / (a + b))
            cumulative_times.append(cumulative_times[-1] + current_time)
            a, b = b, a + b
            phase = 1 - phase
            cycle += 1
        return grid, persistences, phase_lengths, cumulative_times

    def subspace_scan(self, data_grid: np.ndarray, passes: int = 2, use_fib: Optional[bool] = None) -> Tuple[np.ndarray, float]:
        if use_fib is None:
            use_fib = self.use_fibonacci_phases
        current = np.array(data_grid, dtype=float)
        if use_fib:
            final_grid, persistences, _, _ = self.fibonacci_multi_pass_resonance(current)
            return final_grid, persistences[-1]
        else:
            final_pers = 0.0
            for _ in range(passes):
                current, pers, _ = self.dual_pass_resonance(current)
                final_pers = pers
            return current, final_pers

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

    def nested_multi_pass_resonance(self, initial_grid: np.ndarray, depth: int = 2,
                                    use_library: bool = True, use_triad: bool = True) -> Tuple[np.ndarray, List[float]]:
        grid = np.array(initial_grid, dtype=float)
        persistences = []
        for d in range(depth):
            if self.use_fibonacci_phases:
                grid, pers_list, _, _ = self.fibonacci_multi_pass_resonance(grid)
                pers = pers_list[-1] if pers_list else 0.0
            else:
                grid, pers, _ = self.dual_pass_resonance(grid)
            persistences.append(pers)
            if d < depth - 1:
                if use_library:
                    resonance_score = self.query_library_resonance(grid.flatten())
                    grid *= resonance_score
                if use_triad:
                    grid *= self.matter_damping
                    feedback = self.env_feedback_fraction * np.mean(np.abs(grid)) * np.random.randn(*grid.shape)
                    grid += feedback
        final_pers = np.sum(np.abs(grid) > self.prune_threshold) / grid.size
        persistences.append(final_pers)
        return grid, persistences

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
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    ouro = OuroborosFramework()
    test_grid = np.random.uniform(-1, 1, (50, 50))
    print("Base subspace scan persistence:", ouro.subspace_scan(test_grid)[1])
    _, nested_pers = ouro.nested_multi_pass_resonance(test_grid, depth=3)
    print("Nested depth=3 persistences over layers:", nested_pers)
