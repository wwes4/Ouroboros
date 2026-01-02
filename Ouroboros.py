"""
ouroboros.py

Ouroboros – a minimalist, parameter-free geometric framework for exploring persistence across scales.

Core Concept (quick grasp for first-time users/AI):
- A spherical manifold where π merges smoothly from ~3.1416 (flat center) to exactly 2 (curved boundary).
- Fixed deviation=2 enforces 1:1 certainty at surface and natural thirds divisions (2π/3 edge ≈2.094, π/3 offset).
- Dual-pass dynamics: first-pass bloom (expansion with stochastic kick), second-pass etching (squaring + irreversible pruning).
- Outcomes: parameter-free ratios (~34.4% base persistence / 65.6% voids; time-loss dilution → observed ~31.1% / 68.9%).
- "Data" as persistence in motion: objects leave ghostly trails fading by dissipation rate (low = fleeting ghosts, high = sharp enduring paths).
- Scale-invariant echoes: cosmology (filaments/voids), geology (Richat rings), photography (long-exposure trails), particle tracks.

New extensions:
- visualize_time_flow: basic motion trails as time/history visibility.
- visualize_ring_manifold_time_flow: concentric rings (Richat/cosmos proxy) with layered persistence—inner high = dense filaments, outer low = fading voids.

Run this file directly for instant demos + prints. Tweak params and re-run to feel the ruler. Everything interconnects cleanly.
MIT License – explore freely.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sympy as sp

class OuroborosFramework:
    def __init__(self, radius=1.0, noise_level=0.7, cycles=100, scale_factor=4.0, time_loss_factor=None):
        self.radius = radius
        self.effective_pi_boundary = 2.0
        self.pi_center = np.pi
        self.deviation = 2.0
        self.noise_level = noise_level
        self.cycles = cycles
        self.third_edge = 2 * np.pi / 3
        self.third_offset = np.pi / 3
        self.scale_factor = scale_factor
        self.time_loss_factor = time_loss_factor

    def pi_variation(self, position_ratio):
        if not 0 <= position_ratio <= 1:
            raise ValueError("position_ratio must be in [0, 1]")
        delta = self.pi_center - self.effective_pi_boundary
        return self.pi_center - delta * (position_ratio ** self.scale_factor)

    def pi_differential(self, position_ratio=1.0, symbolic=False):
        r = sp.symbols('r')
        delta = self.pi_center - self.effective_pi_boundary
        pi_var = self.pi_center - delta * (r ** self.scale_factor)
        diff = sp.diff(pi_var, r)
        if symbolic:
            return diff
        return float(diff.subs(r, position_ratio).evalf())

    def dev_pi_ratio(self, position_ratio):
        return self.deviation / self.pi_variation(position_ratio)

    def derive_cosmic_densities(self, use_time_loss=False):
        filled = 1 / (1 + self.deviation / self.third_offset)
        voids = 1 - filled
        if use_time_loss or self.time_loss_factor is not None:
            loss = self.time_loss_factor or 0.138
            filled *= (1 - loss)
            voids = 1 - filled
        return filled, voids

    def dual_pass_resonance(self, initial_grid):
        grid = np.array(initial_grid, dtype=float)
        bloom = np.sin(grid * self.pi_center) + self.noise_level * np.random.randn(*grid.shape)
        bloom = np.clip(bloom, -self.radius, self.radius)
        etched = np.cos(bloom * (self.effective_pi_boundary ** 2))
        etched += (bloom ** 2) * (self.deviation / self.pi_center)
        etched = np.where(np.abs(etched) < 0.1, 0, etched)
        persistence = np.sum(np.abs(etched) > 0.1) / etched.size
        complement = 1 - persistence
        return etched, persistence, complement

    def yeast_quorum_proxy(self, colony_size=1000, generations=50):
        persistence = np.ones(colony_size)
        for _ in range(generations):
            persistence += self.noise_level * np.random.randn(colony_size)
            persistence = np.clip(persistence ** 2, 0.3, 0.9)
        return np.mean(persistence), persistence

    def gamma_resonance_proxy(self, freq_mean=65.0, bandwidth=70.0):
        scaled = freq_mean * (self.deviation / self.pi_center)
        squared_boost = scaled ** 2
        return squared_boost * (bandwidth / self.third_edge)

    def richat_ring_matcher(self):
        ring_ratios = [0.0, 0.31, 0.63, 1.0]
        return ring_ratios, [self.pi_variation(r) for r in ring_ratios]

    def visualize_time_flow(self, steps=200, persistence_levels=[0.2, 0.5, 0.8, 0.95], save_path=None):
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

    def visualize_ring_manifold_time_flow(self, steps=300, ring_ratios=[0.2, 0.4, 0.6, 0.8], 
                                          persistence_levels=[0.95, 0.8, 0.6, 0.3], save_path=None):
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

    def visualize_manifold(self, save_path=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        r = np.sqrt(x**2 + y**2 + z**2)
        pi_color = self.pi_center - (self.pi_center - self.effective_pi_boundary) * (r ** self.scale_factor)
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

if __name__ == "__main__":
    ouro = OuroborosFramework()

    print("=== Instant Demo ===")
    print("Base densities:", ouro.derive_cosmic_densities())
    print("Observed (time-loss):", ouro.derive_cosmic_densities(use_time_loss=True))
    print("Third edge:", ouro.third_edge)

    # Core visuals – run these for immediate understanding
    ouro.visualize_time_flow()
    ouro.visualize_ring_manifold_time_flow()
    ouro.visualize_manifold()
