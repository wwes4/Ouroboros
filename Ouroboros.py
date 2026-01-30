"""
Ouroboros.py — Core Geometric Persistence Framework (Slim v8.9 - Fully Updated)
January 16, 2026

Pure mathematical foundation with all core functions restored and verified against prior integrations.
No functions were missed in the final merge — this is the complete, runnable slim core.

Key updates:
- Base frame-rate delta now fully derived (no hardcoded 0.016... value)
- Effective boundary 2.078 remains the sole fixed geometric invariant (minimal asymmetry source)
- Added explicit comments on fractal-to-fractal turbulence and irreversible data loss
- All prior methods (bloom/etch/prune, library feedback, nested passes, consensus, etc.) fully included
- Tanh amplitude stabilization for sustained trails
- Downsample safety, exports, clock, expansion, dynamic CMB all intact

The stack is complete and self-consistent — ready for chains, ticks, and eternal etching.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import json
from observer import MultiObserver
from vibration import propagate_vibration, refract, holographic_linkage

PHI = (1 + np.sqrt(5)) / 2

class OuroborosClock:
    def __init__(self, start_time: Optional[float] = None, time_loss_factor: float = 0.138):
        self.start_tick = 0
        self.current_tick = 0
        self.start_time = start_time or time.time()
        self.time_loss_factor = time_loss_factor
        self.history = []
        self.last_decay = 1.0

    def tick(self):
        self.current_tick += 1
        earthly_now = time.time()
        age = earthly_now - self.start_time
        decay = self.time_loss_factor ** self.current_tick
        self.history.append((self.current_tick, earthly_now, decay))
        self.last_decay = decay
        return decay

    def get_age(self) -> float:
        return time.time() - self.start_time

class OuroborosFramework:
    def __init__(self, use_fibonacci_phases: bool = True,
                 matter_damping: float = 0.98, env_feedback_fraction: float = 0.08,
                 signature_dim: int = 32, max_grid_size: int = 1024):
        self.use_fibonacci_phases = use_fibonacci_phases
        self.matter_damping = matter_damping
        self.env_feedback_fraction = env_feedback_fraction
        self.signature_dim = signature_dim
        self.max_grid_size = max_grid_size
        # Core geometric invariants — fully derived
        self.pi_center = np.pi
        self.theoretical_pi_boundary = 2 * np.pi / 3
        self.third_offset = np.pi / 3
        self.effective_pi_boundary = 2.078 # Sole fixed invariant — minimal asymmetry for unidirectional flow
        self.frame_delta = self.theoretical_pi_boundary - self.effective_pi_boundary # Derived base frame-rate delta ≈ 0.01639510239
        # This delta is the root of fractal-to-fractal turbulence: injects unidirectional bias per cycle → irreversible weak-trail pruning
        self.deviation = self.pi_center - self.effective_pi_boundary # Primary asymmetry strength ≈ 1.06359
        # Matter density proxy (~31% sustain against expansion)
        self.target_filled = 0.31
        self.deviation_matter = (1 / self.target_filled - 1) * self.third_offset
        self.time_loss_factor = 0.138 # Temporal decay proxy
        self.pass_damping = {"physical": 0.995, "wave": 0.95, "data": 0.75}
        self.pass_noise = {"physical": 0.15, "wave": 0.69, "data": 1.5}
        self.truth_library: List[Dict] = []
        # Clock always active
        self.clock = OuroborosClock(time_loss_factor=self.time_loss_factor)
        # Cosmic expansion & dynamic CMB
        self.hubble_tension_low = 67.4 / 30857.0
        self.hubble_tension_high = 73.0 / 30857.0
        self.expansion_factor = (self.hubble_tension_low + self.hubble_tension_high) / 2
        self.prune_threshold = self.expansion_factor * 200 # Derived from cosmic push
        self.initial_cmb = 3000.0 / self.pi_center
        self.current_cmb = self.initial_cmb
        self.scale_factor = 1.0
        # Bootstrap eternal priors
        fib_seq = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55]) / 55.0
        self.add_to_truth_library(fib_seq, "Fibonacci phasing harmonic")
        schumann = np.array([7.83, 14.3, 20.8, 27.3, 33.8]) / 33.8
        self.add_to_truth_library(schumann, "Earth manifold base resonance")
        self.load_truth_library()
        # Communion / agentic extensions
        self.multi_observer = None

        # Wrapped vibration methods (for easy access in chains/agents)
        self.propagate_vibration = lambda amp, dist=10.0, pos=0.5: propagate_vibration(self, amp, dist, pos)
        self.refract = lambda amp, pos=0.5: refract(self, amp, pos)
        self.holographic_linkage = lambda data, pos=0.5, real_freq=None: holographic_linkage(self, data, pos, real_freq)

    def chain(self):
        return DSLChain(self)

    # Core operations
    def _bloom(self, grid: np.ndarray, noise_amp: float) -> np.ndarray:
        expanded = np.sin(grid * self.pi_center) * (1 + noise_amp)
        return expanded + np.random.uniform(-noise_amp, noise_amp, grid.shape)

    def _etch(self, grid: np.ndarray) -> np.ndarray:
        constrained = np.cos(grid) * (grid ** 2 < self.effective_pi_boundary ** 2)
        return constrained + self.deviation * np.tanh(grid / self.pi_center)

    def _prune(self, grid: np.ndarray) -> np.ndarray:
        grid[np.abs(grid) < self.prune_threshold] = 0
        return grid

    def _apply_library_feedback(self, grid: np.ndarray) -> np.ndarray:
        if not self.truth_library:
            return grid
        feedback = np.zeros_like(grid)
        target_len = grid.size
        for item in self.truth_library:
            proj = np.array(item["projected"])
            resampled = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(proj)), proj)
            feedback += resampled.reshape(grid.shape) * self.env_feedback_fraction
        return grid + feedback

    def _downsample_grid(self, grid: np.ndarray) -> np.ndarray:
        if grid.size <= self.max_grid_size:
            return grid
        factor = np.sqrt(grid.size / self.max_grid_size)
        fi = int(np.ceil(factor))
        if grid.ndim > 1:
            new_h, new_w = grid.shape[0] // fi, grid.shape[1] // fi
            down = np.mean(grid.reshape(new_h, fi, new_w, fi), axis=(1,3))
            return down
        else:
            new_size = grid.shape[0] // fi
            return np.mean(grid.reshape(new_size, fi), axis=1)

    def _project_to_signature(self, vec: np.ndarray) -> np.ndarray:
        flat = vec.flatten()
        if len(flat) < self.signature_dim:
            flat = np.pad(flat, (0, self.signature_dim - len(flat)))
        fft_mags = np.abs(np.fft.rfft(flat))
        orig_freq = np.linspace(0, 0.5, len(fft_mags))
        target_freq = np.linspace(0, 0.5, self.signature_dim)
        signature = np.interp(target_freq, orig_freq, fft_mags)
        return signature / (np.linalg.norm(signature) + 1e-8)

    def nested_multi_pass_resonance(self, grid: np.ndarray, depth: int = 2, pass_type: str = "wave") -> tuple[np.ndarray, List[float], float]:
        damping = self.pass_damping[pass_type]
        noise = self.pass_noise[pass_type]
        pers_curve = []
        current = grid.copy()
        for _ in range(depth):
            current = self._bloom(current, noise)
            current = self._etch(current)
            current = self._apply_library_feedback(current)
            current = self._prune(current)
            current *= damping
            current = np.tanh(current) # Amplitude stabilization
            pers = np.count_nonzero(current) / current.size
            pers_curve.append(pers)
        hazard = np.std(pers_curve) if pers_curve else 0.0
        return current, pers_curve, hazard

    def consensus_across_passes(self, grid: np.ndarray, depth: int = 3) -> Dict:
        phys, phys_pers, _ = self.nested_multi_pass_resonance(grid, depth, "physical")
        wave, wave_pers, _ = self.nested_multi_pass_resonance(grid, depth, "wave")
        data, data_pers, _ = self.nested_multi_pass_resonance(grid, depth, "data")
        weights = np.array([np.mean(phys_pers), np.mean(wave_pers), np.mean(data_pers)])
        weights /= (weights.sum() + 1e-8)
        consensus_grid = (weights[0] * phys + weights[1] * wave + weights[2] * data)
        consensus_grid = self._apply_library_feedback(consensus_grid)
        consensus_grid = self._prune(consensus_grid)
        consensus_grid = np.tanh(consensus_grid)
        consensus_pers = np.count_nonzero(consensus_grid) / consensus_grid.size
        return {
            "consensus_grid": consensus_grid,
            "phys_pers": np.mean(phys_pers),
            "wave_pers": np.mean(wave_pers),
            "data_pers": np.mean(data_pers),
            "consensus_pers": consensus_pers,
            "weights": weights.tolist()
        }

    def add_to_truth_library(self, vec: np.ndarray, desc: str):
        proj = self._project_to_signature(vec)
        self.truth_library.append({"projected": proj.tolist(), "desc": desc})
        self.save_truth_library()

    def load_truth_library(self, filename: str = "ouro_truth_library.json"):
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    loaded = json.load(f)
                existing = {item["desc"] for item in self.truth_library}
                added = 0
                for item in loaded:
                    if item["desc"] not in existing:
                        self.truth_library.append(item)
                        existing.add(item["desc"])
                        added += 1
                print(f"Ouroboros: Loaded {added} persistent truths → total {len(self.truth_library)}")
            except Exception as e:
                print(f"Ouroboros: Truth library load failed: {e}")
        else:
            print("Ouroboros: No existing truth library → starting fresh")

    def save_truth_library(self, filename: str = "ouro_truth_library.json"):
        try:
            with open(filename, "w") as f:
                json.dump(self.truth_library, f, indent=4)
            print(f"Ouroboros: Truth library persisted → {len(self.truth_library)} truths")
        except Exception as e:
            print(f"Ouroboros: Truth library save failed: {e}")

    def init_multi_observer(self, num_observers: int = 3, bands: Optional[List[str]] = None):
        """Initialize multi-observer communion system for agentic runs."""
        self.multi_observer = MultiObserver(self, num_observers, bands)
        print(f"Ouroboros: Initialized MultiObserver with {num_observers} agents")
        return self.multi_observer

class DSLChain:
    def __init__(self, framework: OuroborosFramework):
        self.framework = framework
        self.operations = []
        self.clock = framework.clock

    def nested(self, depth: int = 2, pass_type: str = "wave"):
        self.operations.append(("nested", depth, pass_type))
        return self

    def physical(self, depth: int = 2):
        return self.nested(depth, "physical")

    def wave(self, depth: int = 2):
        return self.nested(depth, "wave")

    def data(self, depth: int = 2):
        return self.nested(depth, "data")

    def consensus(self, depth: int = 3):
        self.operations.append(("consensus", depth))
        return self

    def tick(self, count: int = 1):
        self.operations.append(("tick", count))
        return self

    def export_json(self, filename: str = "chain_result.json"):
        self.operations.append(("export_json", filename))
        return self

    def export_npz(self, filename: str = "chain_result.npz"):
        self.operations.append(("export_npz", filename))
        return self

    def run(self, initial_grid: np.ndarray) -> Dict:
        grid = self.framework._downsample_grid(initial_grid.copy())
        results = {"final_grid": grid, "history": [], "exports": []}
        for op in self.operations:
            if op[0] == "tick":
                for _ in range(op[1]):
                    self.clock.tick()
                    self.framework.scale_factor = 1.0 + (self.framework.expansion_factor * self.clock.current_tick)
                    self.framework.current_cmb = self.framework.initial_cmb / self.framework.scale_factor
                    # Gentle uniform floor to prevent total cold prune
                    grid += self.framework.current_cmb * 0.01
                    # Expansion stretch
                    grid += self.framework.expansion_factor * np.sign(grid)
            elif op[0] == "nested":
                grid, pers_list, hazard = self.framework.nested_multi_pass_resonance(
                    grid, depth=op[1], pass_type=op[2]
                )
                results["history"].append({
                    "type": "nested",
                    "pass": op[2],
                    "pers_curve": pers_list,
                    "hazard": hazard
                })
            elif op[0] == "consensus":
                cons = self.framework.consensus_across_passes(grid, depth=op[1])
                grid = cons["consensus_grid"]
                results["history"].append({
                    "type": "consensus",
                    "details": cons
                })
            elif op[0] == "export_json":
                data = {
                    "final_pers": results["history"][-1].get("pers_curve", [0])[-1] if results["history"] else 0.0,
                    "consensus_pers": results["history"][-1]["details"]["consensus_pers"] if results["history"] and "details" in results["history"][-1] else 0.0,
                    "grid": grid.flatten().tolist(),
                    "history": results["history"],
                    "clock": {"ticks": self.clock.current_tick, "cmb": self.framework.current_cmb}
                }
                with open(op[1], "w") as f:
                    json.dump(data, f, indent=4)
                results["exports"].append(op[1])
            elif op[0] == "export_npz":
                np.savez_compressed(op[1], grid=grid, history=results["history"])
                results["exports"].append(op[1])
        results["final_grid"] = grid
        return results

if __name__ == "__main__":
    ouro = OuroborosFramework(use_fibonacci_phases=True)
    print(f"\nOuroboros v8.9 slim core initialized — {len(ouro.truth_library)} truths active")
    print(f"Baseline real-world time captured: {time.ctime(ouro.clock.start_time)}")
    print(f"Initial CMB proxy (hot): {ouro.current_cmb:.4f} | Expansion factor: {ouro.expansion_factor:.8f}")
    print(f"Base frame-rate delta (derived): {ouro.frame_delta:.14f}")

    test_grid = np.random.uniform(-1, 1, (32, 32))
    result = ouro.chain().physical(3).wave(4).data(3).consensus(3).run(test_grid)
    final_pers = result["history"][-1]["details"]["consensus_pers"]
    print(f"No-tick chain persistence: {final_pers:.4f}")

    result_time = ouro.chain().wave(5).tick(50).physical(3).consensus(3).run(test_grid)
    print(f"After 50 ticks — persistence: {result_time['history'][-1]['details']['consensus_pers']:.4f}")
    print(f"Current dynamic CMB floor: {ouro.current_cmb:.4f}")
    print(f"Manifold ticks: {ouro.clock.current_tick}")