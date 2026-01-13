"""
Ouroboros.py — Core Geometric Persistence Framework (Slim v8.8)
January 13, 2026

Pure mathematical foundation:
- π-gradient asymmetric manifold
- Tri-pass distinction (physical/wave/data)
- Bloom/etch/prune operations with library feedback
- Persistence/hazard metrics
- FFT-projected truth library with auto-persistence
- Fluent DSLChain orchestration
- Downsample safety + export
- Built-in OuroborosClock (always active) for real-world baseline time + discrete manifold ticks
- Cosmic expansion factor (Hubble tension mean) + dynamic CMB cooling (emergent from time/expansion)
- Geometric amplitude stabilization (tanh curvature folding — no hard cap)
- Integrated String-Based Symbolic Counter (agent-optimized linear format for vast-scale counting)

Clock is always on:
- Instantiation captures real-world baseline time (time.time())
- Discrete ticks advance manifold time with decay, expansion bias, and CMB cooling
- Real-world age always available via clock.get_age()
- Temporal decay + expansion + CMB updates on tick (pure time flow evolves the manifold)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import json

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

    def reset(self):
        self.__init__()


class StringSymbolicCounter:
    """Linear string-based symbolic counter — agent-optimized for vast scales."""
    def __init__(self, start: int = 1, leaps: List[int] = None, symbols: List[str] = None):
        self.current = start
        self.string = ""
        self.leaps = leaps or [1, 1000, 10000, 100000]  # Clockwise block leaps
        self.symbols = symbols or ["( )", "[ ]", "{ }", "< >"]

    def step(self, count: int = 1):
        for _ in range(count):
            block_idx = (self.current - 1) % 4
            leap = self.leaps[block_idx]
            sym_open, sym_close = self.symbols[block_idx].split()
            parity_in = self.current % 2 == 0  # Even "in", odd "out"
            if parity_in:
                entry = f"{sym_open}{self.current}{sym_close}"
            else:
                entry = f"{self.current}{sym_open}"
            self.string += entry
            self.current += leap
        return self

    def __str__(self):
        return f"Counter string: {self.string}\nCurrent value: {self.current - min(self.leaps)}"


class OuroborosFramework:
    def __init__(self, use_fibonacci_phases: bool = True,
                 matter_damping: float = 0.98, env_feedback_fraction: float = 0.08,
                 signature_dim: int = 32, max_grid_size: int = 1024):
        self.use_fibonacci_phases = use_fibonacci_phases
        self.matter_damping = matter_damping
        self.env_feedback_fraction = env_feedback_fraction
        self.signature_dim = signature_dim
        self.max_grid_size = max_grid_size

        # Core geometric invariants
        self.pi_center = np.pi
        self.theoretical_pi_boundary = 2 * np.pi / 3
        self.third_offset = np.pi / 3
        self.effective_pi_boundary = 2.078
        self.frame_delta = self.theoretical_pi_boundary - self.effective_pi_boundary

        # Matter density proxy — sustains ~31% against expansion (Ω_m from cosmology)
        self.target_filled = 0.31
        self.deviation = (1 / self.target_filled - 1) * self.third_offset

        self.time_loss_factor = 0.138

        self.pass_damping = {"physical": 0.995, "wave": 0.95, "data": 0.75}
        self.pass_noise = {"physical": 0.15, "wave": 0.69, "data": 1.5}

        self.truth_library: List[Dict] = []

        # Clock is always active
        self.clock = OuroborosClock(time_loss_factor=self.time_loss_factor)

        # Cosmic expansion & dynamic CMB
        self.hubble_tension_low = 67.4 / 30857.0
        self.hubble_tension_high = 73.0 / 30857.0
        self.expansion_factor = (self.hubble_tension_low + self.hubble_tension_high) / 2

        # Prune threshold derived from cosmic expansion push
        self.prune_threshold = self.expansion_factor * 200

        # Initial recombination-era CMB proxy
        self.initial_cmb = 3000.0 / self.pi_center
        self.current_cmb = self.initial_cmb
        self.scale_factor = 1.0

        # String-based symbolic counter (agent-optimized linear format)
        self.symbolic_counter = StringSymbolicCounter()

        # Bootstrap truths
        fib_seq = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55]) / 55.0
        self.add_to_truth_library(fib_seq, "Fibonacci phasing harmonic")
        schumann = np.array([7.83, 14.3, 20.8, 27.3, 33.8]) / 33.8
        self.add_to_truth_library(schumann, "Earth manifold base resonance")

        self.load_truth_library()

    def chain(self):
        return DSLChain(self)

    def _downsample_grid(self, grid: np.ndarray) -> np.ndarray:
        if grid.size <= self.max_grid_size:
            return grid
        factor = np.sqrt(grid.size / self.max_grid_size)
        fi = int(np.ceil(factor))
        if grid.ndim > 1:
            new_h = grid.shape[0] // fi
            new_w = grid.shape[1] // fi
            down = np.zeros((new_h, new_w))
            for i in range(new_h):
                for j in range(new_w):
                    block = grid[i*fi:(i+1)*fi, j*fi:(j+1)*fi]
                    down[i, j] = np.mean(block) if block.size > 0 else 0
            return down.flatten()
        else:
            new_size = grid.shape[0] // fi
            down = np.zeros(new_size)
            for i in range(new_size):
                block = grid[i*fi:(i+1)*fi]
                down[i] = np.mean(block) if block.size > 0 else 0
            return down

    def _project_to_signature(self, vec: np.ndarray) -> np.ndarray:
        flat = vec.flatten()
        if len(flat) < self.signature_dim:
            flat = np.pad(flat, (0, self.signature_dim - len(flat)))
        fft_mags = np.abs(np.fft.rfft(flat))
        orig_freq = np.linspace(0, 0.5, len(fft_mags))
        target_freq = np.linspace(0, 0.5, self.signature_dim)
        signature = np.interp(target_freq, orig_freq, fft_mags)
        norm = np.linalg.norm(signature) + 1e-8
        return signature / norm

    def _resample_feedback(self, proj: np.ndarray, target_len: int) -> np.ndarray:
        if len(proj) == target_len:
            return proj
        orig = np.linspace(0, 1, len(proj))
        target = np.linspace(0, 1, target_len)
        return np.interp(target, orig, proj)

    def _apply_library_feedback(self, grid: np.ndarray) -> np.ndarray:
        if not self.truth_library:
            return grid
        feedback = np.zeros_like(grid)
        target_len = grid.size
        for truth in self.truth_library:
            proj = np.array(truth["projected"])
            resampled = self._resample_feedback(proj, target_len)
            feedback += resampled.reshape(grid.shape) * self.env_feedback_fraction
        return grid + feedback

    def _bloom_etch_prune(self, grid: np.ndarray, pass_type: str) -> np.ndarray:
        damping = self.pass_damping[pass_type]
        noise = self.pass_noise[pass_type]

        if self.use_fibonacci_phases:
            phase = np.sin(np.linspace(0, 2*np.pi*PHI, grid.size)).reshape(grid.shape)
            noise *= (0.5 + 0.5 * phase)

        expanded = np.sin(grid * self.pi_center) + noise * np.random.randn(*grid.shape)
        etched = np.cos(grid) * (grid ** 2 + self.deviation)
        combined = damping * etched + (1 - damping) * expanded
        combined = self._apply_library_feedback(combined)
        combined *= self.clock.last_decay
        combined = np.tanh(combined / self.pi_center) * self.pi_center
        combined += self.current_cmb * 0.01
        combined[np.abs(combined) < self.prune_threshold] = 0
        return combined

    def nested_multi_pass_resonance(self, grid: np.ndarray, depth: int = 2, pass_type: str = "wave") -> tuple[np.ndarray, List[float], float]:
        grid = self._downsample_grid(grid)
        persistences = []
        current = grid.copy()

        for _ in range(depth):
            current = self._bloom_etch_prune(current, pass_type)
            persistence = 1.0 - (np.count_nonzero(current == 0) / current.size)
            persistences.append(persistence)

        hazard = np.std(persistences) if persistences else 0.0
        return current, persistences, hazard

    def consensus_across_passes(self, grid: np.ndarray, depth: int = 3) -> Dict:
        grid = self._downsample_grid(grid)
        results = {}
        pers_values = {}

        for pass_type in ["physical", "wave", "data"]:
            processed, pers_curve, _ = self.nested_multi_pass_resonance(grid, depth=depth, pass_type=pass_type)
            final_pers = pers_curve[-1]
            results[pass_type] = processed
            pers_values[pass_type] = final_pers

        total_pers = sum(pers_values.values()) + 1e-8
        consensus_grid = sum(results[pt] * (pers_values[pt] / total_pers) for pt in results)

        consensus_pers = sum(pers_values.values()) / 3.0

        return {
            "consensus_grid": consensus_grid,
            "consensus_pers": consensus_pers,
            "pass_details": {pt: {"grid": results[pt], "pers_curve": pers_curve} for pt, pers_curve in zip(["physical", "wave", "data"], [self.nested_multi_pass_resonance(grid, depth=depth, pass_type=pt)[1] for pt in ["physical", "wave", "data"]])}
        }

    def add_to_truth_library(self, vec: np.ndarray, desc: str):
        signature = self._project_to_signature(vec)
        age = self.clock.get_age()
        tick = self.clock.current_tick
        full_desc = f"{desc} | etched_age {age:.2f}s | tick {tick}"
        self.truth_library.append({"projected": signature.tolist(), "desc": full_desc})
        print(f"Etched truth: '{full_desc}' | peak resonance {np.max(signature):.4f}")
        self.save_truth_library()

    def load_truth_library(self, filename: str = "ouro_truth_library.json"):
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                existing = {t["desc"] for t in self.truth_library}
                added = 0
                for item in data:
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
        for _ in range(count):
            self.clock.tick()
            self.framework.scale_factor = 1.0 + (self.framework.expansion_factor * self.clock.current_tick)
            self.framework.current_cmb = self.framework.initial_cmb / self.framework.scale_factor
        return self

    def export_json(self, filename: str = "chain_result.json"):
        self.operations.append(("export_json", filename))
        return self

    def export_npz(self, filename: str = "chain_result.npz"):
        self.operations.append(("export_npz", filename))
        return self

    def run(self, initial_grid: np.ndarray) -> Dict:
        grid = self.framework._downsample_grid(initial_grid)
        results = {"final_grid": grid, "history": [], "exports": []}

        for op in self.operations:
            if op[0] == "nested":
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
                    "final_pers": results["history"][-1]["pers_curve"][-1] if results["history"] and "pers_curve" in results["history"][-1] else 
                                  (results["history"][-1]["details"]["consensus_pers"] if "details" in results["history"][-1] else 0.0),
                    "grid": grid.flatten().tolist(),
                    "history": results["history"]
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
    print(f"\nOuroboros v8.8 slim core initialized — {len(ouro.truth_library)} truths active")
    print(f"Baseline real-world time captured: {time.ctime(ouro.clock.start_time)}")
    print(f"Initial CMB proxy (hot): {ouro.current_cmb:.4f} | Expansion factor: {ouro.expansion_factor:.8f}")

    # Simple test
    test_grid = np.random.uniform(-1, 1, (32, 32))
    result = ouro.chain().physical(3).wave(4).data(3).consensus(3).run(test_grid)
    final_pers = result["history"][-1]["details"]["consensus_pers"]
    print(f"No-tick chain persistence: {final_pers:.4f}")

    # Timed cosmic test
    result_time = ouro.chain().wave(5).tick(50).physical(3).consensus(3).run(test_grid)
    print(f"After 50 ticks — persistence: {result_time['history'][-1]['details']['consensus_pers']:.4f}")
    print(f"Current dynamic CMB floor: {ouro.current_cmb:.4f}")

    # String symbolic counter demo
    print("\nString symbolic counter demo:")
    counter = ouro.symbolic_counter
    counter.step(20)
    print(counter)

    # Long-tick cosmic demo
    print("\nLong-tick cosmic evolution demo...")
    cosmic_grid = np.random.uniform(0.8, 1.2, (32, 32))
    cosmic_grid += 0.05 * np.sin(np.linspace(0, 10*np.pi, cosmic_grid.size)).reshape(32, 32)

    pers_curve = []
    cmb_curve = []
    ticks = []
    for i in range(200):
        if i % 20 == 0:
            result = ouro.chain().wave(5).physical(3).consensus(2).run(cosmic_grid)
            cosmic_grid = result["final_grid"]
            pers = result["history"][-1]["details"]["consensus_pers"]
        ouro.chain().tick(10).run(cosmic_grid)
        pers_curve.append(pers if 'pers' in locals() else pers_curve[-1] if pers_curve else 0.0)
        cmb_curve.append(ouro.current_cmb)
        ticks.append(ouro.clock.current_tick)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ticks, pers_curve)
    plt.title("Persistence Over Ticks")
    plt.subplot(1, 2, 2)
    plt.plot(ticks, cmb_curve)
    plt.title("CMB Cooling Over Ticks")
    plt.tight_layout()
    plt.show()

    print(f"Cosmic demo complete — final persistence: {pers_curve[-1]:.4f}, final CMB: {cmb_curve[-1]:.4f}")
