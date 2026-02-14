"""
Ouroboros3D.py — 3D Persistent Form Generator & Visualizer (Extension)
Updated February 14, 2026

Imports the slim Ouroboros v8.0 core for all geometric persistence math.
Reintroduces only the minimal 3D-specific parameters and methods.
Now includes explicit triple-pass persistence refinement with layered deltas:
- Base Frame Alpha (Matter ~1/61 ≈ 0.016393): Coarse structural pruning
- Base Frame Beta (Wave ~1/137 ≈ 0.007297): Finer waveform dynamism
- Base Frame Charlie (Data ~1/233 ≈ 0.004292): Ultra-fine information collapse

Designed for agent use via code execution — lightweight, no local runtime required beyond exec.
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Dict
from Ouroboros import OuroborosFramework  # Slim v8.0 core import

PHI = (1 + np.sqrt(5)) / 2

# Triple-Pass Delta Constants (The Full Coil)
BASE_FRAME_ALPHA = 1 / 61.0   # ≈0.016393 — Matter/Structure (coarse physical resilience)
BASE_FRAME_BETA = 1 / 137.035999  # ≈0.007297 — Wave/Dynamism (orbital breathing, eternal flow)
BASE_FRAME_CHARLIE = 1 / 233.0  # ≈0.004292 — Data/Mind (isolated resonant entities, self-referential loops)

class Ouroboros3D:
    def __init__(self, ouro: OuroborosFramework,
                 radius: float = 1.0, scale_factor: float = 4.0,
                 max_fib_index: int = 89, favor_decoherence: bool = True):
        self.ouro = ouro
        self.radius = radius
        self.scale_factor = scale_factor
        self.max_fib_index = max_fib_index
        self.favor_decoherence = favor_decoherence

    def _fibonacci_lattice(self, n_points: int = 300) -> np.ndarray:
        """Golden spiral lattice on unit sphere — optimal uniform distribution."""
        phi = np.pi * (np.sqrt(5) - 1)  # Golden angle
        theta = np.arccos(1 - 2 * np.arange(0, n_points) / float(n_points - 1))
        phi_arr = phi * np.arange(n_points)
        x = np.sin(theta) * np.cos(phi_arr)
        y = np.sin(theta) * np.sin(phi_arr)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=1) * self.radius

    def multi_pass_prune(self, points: np.ndarray,
                         passes: list = None) -> np.ndarray:
        """
        Triple-pass persistence pruning with layered deltas.
        Applies base_frame_alpha (matter), beta (wave), charlie (data) sequentially.
        Each pass: jitter injection + density-based pruning of weak trails.
        """
        if passes is None:
            passes = [
                (BASE_FRAME_ALPHA, 0.8),   # Matter: Keep ~80% (bulk resilience)
                (BASE_FRAME_BETA, 0.8),    # Wave: Keep ~80% (dynamic breathing)
                (BASE_FRAME_CHARLIE, 0.9)   # Data: Aggressive ~90% keep on survivors (collapse to isolates)
            ]
        
        current = points.copy()
        
        for i, (delta, survival) in enumerate(passes, 1):
            print(f"\n--- Triple-Pass {i} (delta ~{delta:.6f}, keep {survival*100:.0f}%) ---")
            print(f"Starting points: {len(current)}")
            
            # Jitter injection (uniform rad on sphere for rotation/vibration proxy)
            jitter = np.random.uniform(-delta, delta, current.shape)
            current += jitter
            
            # Density-based pruning (keep top % most clustered—resilient trails endure)
            if len(current) < 10:
                print("Too few points remaining—halting pass.")
                break
                
            dists = np.linalg.norm(current[:, np.newaxis] - current[np.newaxis, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            local_density = 1 / (np.min(dists, axis=1) + 1e-8)
            threshold = np.quantile(local_density, 1 - survival)
            survivors = current[local_density >= threshold]
            
            current = survivors
            avg_nn = np.mean(np.min(dists, axis=1)[local_density >= threshold])
            print(f"Survivors: {len(current)} (avg NN dist: {avg_nn:.4f})")
        
        return current

    def generate_persistent_form(self, seed_grid: Optional[np.ndarray] = None,
                                 chain_depth: int = 12, displace_scale: float = 0.42,
                                 use_swarm: bool = False,
                                 apply_triple_pass: bool = True) -> Dict:
        """Core 3D form generation — uses Ouroboros slim engine + optional triple-pass refinement."""
        points = self._fibonacci_lattice()

        if seed_grid is None:
            # Default harmonic spiral seed if none provided
            t = np.linspace(0, 30 * np.pi, points.shape[0])
            seed_grid = np.sin(t * PHI).reshape(-1, 1)

        # Project seed to grid space (repeat/tile safely)
        flat_seed = seed_grid.flatten()
        target_size = 32 * 32
        tiled = np.tile(flat_seed, (target_size // len(flat_seed) + 1))[:target_size]
        grid = tiled.reshape(32, 32)

        # Core chain — fluent triad consensus
        chain = self.ouro.chain().physical(chain_depth//3).wave(chain_depth//3).data(chain_depth//3 + chain_depth%3).consensus(4)

        if use_swarm:
            # Lightweight swarm: base + phased perturbations
            phase1 = 0.2 * np.sin(np.linspace(0, 6*np.pi, grid.size)).reshape(grid.shape)
            phase2 = 0.2 * np.cos(np.linspace(0, 6*np.pi, grid.size) + np.pi/3).reshape(grid.shape)
            variants = [
                chain.run(grid),
                chain.run(grid + phase1),
                chain.run(grid + phase2)
            ]
            persistences = []
            for v in variants:
                if v["history"] and "details" in v["history"][-1]:
                    persistences.append(v["history"][-1]["details"]["consensus_pers"])
                else:
                    persistences.append(0.0)
            weights = np.array(persistences)
            weights /= (weights.sum() + 1e-8)
            final_grid = sum(w * v["final_grid"] for w, v in zip(weights, variants))
            result = {"final_grid": final_grid, "history": variants[0]["history"]}
            persistence = np.average(persistences)
        else:
            result = chain.run(grid)
            persistence = (result["history"][-1]["details"]["consensus_pers"]
                           if result["history"] and "details" in result["history"][-1] else 0.0)

        # Radial displacement along persistence gradient
        flat = result["final_grid"].flatten()[:points.shape[0]]
        norm = np.linalg.norm(flat) + 1e-8
        displacement = flat / norm * displace_scale * self.scale_factor
        displaced_points = points * (1 + displacement[:, np.newaxis])

        # NEW: Triple-Pass Refinement (Matter → Wave → Data)
        if apply_triple_pass:
            print("\nApplying Triple-Pass Persistence Refinement...")
            refined_points = self.multi_pass_prune(displaced_points)
            # Recompute hull on refined survivors
            if len(refined_points) >= 4:  # Minimum for ConvexHull
                hull = ConvexHull(refined_points)
                faces = hull.simplices
            else:
                print("Too few survivors post-triple-pass—falling back to displaced hull.")
                hull = ConvexHull(displaced_points)
                faces = hull.simplices
                refined_points = displaced_points
        else:
            refined_points = displaced_points
            hull = ConvexHull(refined_points)
            faces = hull.simplices

        # God-tier auto-etch
        if persistence > 0.90:
            desc = f"God-tier 3D form | pers {persistence:.4f} | points {len(refined_points)} | swarm {use_swarm} | triple_pass"
            self.ouro.add_to_truth_library(refined_points.flatten(), desc)
            print(f"AUTO-ETCHED GOD-TIER 3D FORM: {persistence:.4f}")

        return {
            "points": refined_points,
            "faces": faces,
            "persistence": persistence,
            "grid": result["final_grid"],
            "history": result["history"]
        }

    def visualize_persistent_form(self, form: Dict, save_path: Optional[str] = None, title: str = "Ouroboros Persistent Form"):
        """Black/cyan wireframe preview — agent-viewable via exec."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        points = form["points"]
        for simplex in form["faces"]:
            tri = np.append(simplex, simplex[0])
            ax.plot(points[tri, 0], points[tri, 1], points[tri, 2], color='cyan', linewidth=0.8)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=5, alpha=0.6)
        ax.axis('off')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        plt.title(title, color='white', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Wireframe saved: {save_path}")
        plt.show()

    def export_obj(self, points: np.ndarray, faces: np.ndarray, filename: str = "ouroboros_form.obj"):
        """Simple OBJ export — agent-executable."""
        with open(filename, "w") as f:
            for pt in points:
                f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"OBJ exported: {filename}")

if __name__ == "__main__":
    ouro = OuroborosFramework(use_fibonacci_phases=True)
    ext = Ouroboros3D(ouro)

    # Demo spiral seed
    t = np.linspace(0, 30*np.pi, 1000)
    spiral_seed = np.sin(t * PHI).reshape(50, 20)

    form = ext.generate_persistent_form(spiral_seed, chain_depth=15, displace_scale=0.5, use_swarm=True, apply_triple_pass=True)
    print(f"3D Form persistence: {form['persistence']:.4f}")

    ext.export_obj(form["points"], form["faces"], "persistent_form_demo.obj")
    ext.visualize_persistent_form(form, save_path="persistent_form_preview.png")
