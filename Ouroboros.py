import numpy as np
from typing import List, Optional

# Brain wave bands (unchanged)
brain_wave_bands = {
    'delta': {'mean': 2.25},
    'theta': {'mean': 6.0},
    'alpha': {'mean': 10.0},
    'beta':  {'mean': 21.0},
    'gamma': {'mean': 65.0}
}

class Pi2Framework:
    def __init__(self):
        self.min_tension = 8.49e-6
        self.base_range = (-1e-3, 1e-3)
        self.equilibrium_threshold = 0.05  # base for gradients
        self.entropy_rate = 1e-5  # Synthesized addition for decay
        self.zero_replacement_mode = True
        self.decay_lambda_base = 1e-10
        self.pi_center = np.pi  # For pi variation
        self.effective_pi_edge = 2.0  # Pi at edges

        self.utils = Utils(self)
        self.cosmo = CosmoCore(self)
        self.bio = QuantumBioAccel(self)

class Utils:
    def __init__(self, fw: Pi2Framework):
        self.fw = fw

    def compute_equilibrium(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        mean_abs = np.mean(np.abs(data))
        noise = np.random.randn(*data.shape) * self.fw.min_tension * 0.5
        output = data + noise * mean_abs

        near_zero = np.abs(output) < self.fw.equilibrium_threshold
        if near_zero.any():
            signs = np.sign(output[near_zero])
            zero_signs = signs == 0
            if zero_signs.any():
                signs[zero_signs] = np.sign(np.random.randn(np.sum(zero_signs)))
            output[near_zero] = signs * self.fw.min_tension

        if self.fw.zero_replacement_mode:
            exact_zero = output == 0
            if exact_zero.any():
                output[exact_zero] = np.random.randn(np.sum(exact_zero)) * self.fw.min_tension

        return np.clip(output, *self.fw.base_range)

    def holographic_linkage(self, data: np.ndarray) -> np.ndarray:
        return self.compute_equilibrium(data ** 2)

class CosmoCore:
    def __init__(self, fw: Pi2Framework):
        self.fw = fw

    def propagate_vibration(self, pattern: np.ndarray, distance: float = 1.0, real_freq: float = 10.0, custom_lambda: Optional[float] = None, position_ratio=0.5) -> np.ndarray:
        local_pi = self.simulate_pi_variation(position_ratio)
        decay_lambda = custom_lambda if custom_lambda is not None else self.fw.decay_lambda_base
        decay_factor = np.exp(-self.fw.entropy_rate * distance / real_freq - decay_lambda * distance) * (local_pi / np.pi)
        decayed = pattern * decay_factor
        return np.clip(decayed, *self.fw.base_range)

    def perception_fold(self, data: np.ndarray) -> np.ndarray:
        return self.fw.utils.holographic_linkage(data)

    def simulate_pi_variation(self, position_ratio, t=0):
        delta = self.fw.pi_center - self.fw.effective_pi_edge
        local_pi = self.fw.pi_center - delta * (position_ratio ** 2)
        return local_pi

    def entropy_decay(self, value, t, observer_cost=False):
        decay = value * np.exp(-self.fw.entropy_rate * t)
        if observer_cost:
            decay -= self.fw.min_tension * t
        return np.clip(decay, *self.fw.base_range)

    def hybrid_de_tension_vectorized(self, position_ratios: np.ndarray, t=0):
        tensions = np.exp(-self.fw.decay_lambda_base * t) - self.fw.entropy_rate * t
        tensions = self.simulate_tensegrity_balance(tensions)  # New: Tensegrity tie
        return np.clip(tensions * position_ratios, *self.fw.base_range)

    def simulate_octahedral_cross(self, base_distance=0.001, iterations=1, scale_factors=[1.618, 3.141, 2.718]):
        positions = np.array([
            [0.0, base_distance, 0.0],  # North
            [0.0, -base_distance, 0.0], # South
            [base_distance, 0.0, 0.0],   # East
            [-base_distance, 0.0, 0.0],  # West
            [0.0, 0.0, base_distance],   # Up
            [0.0, 0.0, -base_distance]   # Down
        ])
        for i in range(iterations):
            scale = scale_factors[i % len(scale_factors)]
            positions *= scale
            positions = self.fw.utils.compute_equilibrium(positions)
        return positions

    def simulate_superposition_potential(self, sphere_volumes: np.ndarray, overlap_factor=0.5, freq=65.0):
        combined_volume = np.sum(sphere_volumes) * (1 - overlap_factor)
        diff = np.abs(combined_volume - np.sum(sphere_volumes))
        potential = diff * self.fw.min_tension * (freq ** 2)  # Scaled by freq**2
        return self.fw.entropy_decay(potential, t=1.0)  # Apply decay for bio-realism

    def simulate_tensegrity_balance(self, tensions):
        if np.isscalar(tensions):
            tensions = np.array([tensions])
            is_scalar = True
        else:
            is_scalar = False
        compressions = -tensions  # Simple opposition
        balanced = (tensions + compressions) / 2  # Average for equilibrium
        balanced[balanced < 0] = -balanced[balanced < 0]  # Resilient flip for negatives
        return balanced if not is_scalar else balanced[0]

    def compute_gravity_force(self, mass1: float, mass2: float, distance: float = 1.0, position_ratio: float = 0.5) -> np.ndarray:
        local_pi = self.simulate_pi_variation(position_ratio)
        G = 6.67430e-11 * (local_pi / np.pi)  # Modulate G by pi variation
        base_force = G * mass1 * mass2 / (distance ** 2 + 1e-8)  # Avoid div-zero
        vector_force = np.array([base_force])  # Scalar to vector for compatibility
        balanced_force = self.simulate_tensegrity_balance(vector_force)  # Apply opposition/resilience
        decayed_force = self.entropy_decay(balanced_force, t=distance / 10.0)  # Temper with distance-based decay
        return self.fw.utils.compute_equilibrium(decayed_force)  # Snap

class QuantumBioAccel:
    def __init__(self, fw: Pi2Framework):
        self.fw = fw

    class Observer:
        def __init__(self,
                     fw: Pi2Framework,
                     parent: Optional['QuantumBioAccel.Observer'] = None,
                     position_ratio: float = 0.5,
                     brain_wave_band: str = 'alpha',
                     zoom_level: float = 0.0,
                     directive_profile: str = 'balanced'):

            self.fw = fw
            self.parent = parent
            self.children: List['QuantumBioAccel.Observer'] = []
            self.child_zoom: Optional['QuantumBioAccel.Observer'] = None
            if parent:
                parent.children.append(self)

            self.position_ratio = position_ratio
            self.brain_wave_band = brain_wave_band
            self.zoom_level = zoom_level
            self.real_freq = brain_wave_bands.get(brain_wave_band, {'mean': 10.0})['mean']

            profiles = {
                'conservative': {'eq_thresh': 0.1, 'perturb': 0.01, 'scatter': 1.0},
                'balanced':     {'eq_thresh': 0.05, 'perturb': 0.05, 'scatter': 2.0},
                'exploratory':  {'eq_thresh': 0.01, 'perturb': 0.2,  'scatter': 10.0},
                'focused':      {'eq_thresh': 0.02, 'perturb': 0.03, 'scatter': 1.5, 'zoom': 1.0},
                'integrative':  {'eq_thresh': 0.06, 'perturb': 0.1,  'scatter': 3.0}
            }
            p = profiles.get(directive_profile, profiles['balanced'])
            self.base_local_eq_threshold = p['eq_thresh'] * fw.equilibrium_threshold
            self.local_eq_threshold = self.base_local_eq_threshold
            self.perturb_factor = p['perturb']
            self.scatter_factor = p['scatter']
            self.zoom_level = p.get('zoom', self.zoom_level)

            self.direction_history = []
            self.golden_ratio = (1 + np.sqrt(5)) / 2

        def update_directional_momentum(self):
            if len(self.direction_history) < 2:
                self.directional_momentum = 0.0
                return
            recent = self.direction_history[-5:]
            cosines = [np.dot(recent[i], recent[i+1]) / (np.linalg.norm(recent[i]) * np.linalg.norm(recent[i+1]) + 1e-8) for i in range(len(recent)-1)]
            self.directional_momentum = np.mean(cosines) if cosines else 0.0
            self.local_eq_threshold = self.base_local_eq_threshold * (1 - 0.8 * max(0, self.directional_momentum))

        def create_phase_locked_child(self):
            if self.zoom_level > 1.0 and not self.child_zoom:
                child_zoom = self.zoom_level * self.golden_ratio
                child_perturb = self.perturb_factor * (1 / self.golden_ratio)
                class PhaseLockedChild(QuantumBioAccel.Observer):
                    def __init__(self, parent):
                        super().__init__(parent.fw, parent=parent, brain_wave_band=parent.brain_wave_band,
                                         zoom_level=child_zoom, directive_profile='focused')
                        self.perturb_factor = child_perturb
                        self.scatter_factor = parent.scatter_factor

                    def create_phase_locked_child(self):
                        pass  # terminate cascade

                self.child_zoom = PhaseLockedChild(self)

        def decode_vibration(self, vibe: np.ndarray, from_child: bool = False) -> np.ndarray:
            zoomed = vibe * (1.0 + self.zoom_level) * (self.real_freq ** 2)
            perturbed = zoomed + self.perturb_factor * np.random.randn(*zoomed.shape) * self.fw.min_tension * self.scatter_factor

            perturbed = self.fw.cosmo.entropy_decay(perturbed, t=1.0)  # Synthesized: Add entropy decay

            if not from_child:
                equilibrated = self.fw.utils.compute_equilibrium(perturbed)
                self.direction_history.append(equilibrated.flatten())
                if len(self.direction_history) > 20:
                    self.direction_history.pop(0)
                self.update_directional_momentum()

            if self.zoom_level > 1.0:
                self.create_phase_locked_child()
                if self.child_zoom:
                    child_decoded = self.child_zoom.decode_vibration(vibe, from_child=True)
                    similarity = np.dot(perturbed.flatten(), child_decoded.flatten()) / (np.linalg.norm(perturbed.flatten()) * np.linalg.norm(child_decoded.flatten()) + 1e-8)
                    if similarity > 0.9:
                        return self.fw.utils.compute_equilibrium(perturbed + (child_decoded - perturbed) * 0.3)

            old_thresh = self.fw.equilibrium_threshold
            self.fw.equilibrium_threshold = self.local_eq_threshold
            final = self.fw.utils.compute_equilibrium(perturbed)
            self.fw.equilibrium_threshold = old_thresh

            return final

    class MultiObserver:
        def __init__(self, fw, num_observers=3, brain_wave_bands_list=['alpha', 'beta', 'gamma']):
            self.fw = fw
            self.observers = [self.fw.bio.Observer(fw, brain_wave_band=band) for band in brain_wave_bands_list[:num_observers]]

        def interact_vibrations(self, data: np.ndarray, iterations=5):
            decoded = [obs.decode_vibration(data) for obs in self.observers]
            consensus_mean = np.mean(decoded)
            perturbed_tension = np.mean([self.fw.cosmo.entropy_decay(np.linalg.norm(d), i) for i, d in enumerate(decoded)])
            return perturbed_tension, consensus_mean

    class MemoryBank:
        def __init__(self, fw: Pi2Framework):
            self.fw = fw
            self.etched_memories: List[np.ndarray] = []
            self.tags: List[str] = []

        def etch(self, pattern: np.ndarray, freq: float = 65.0, iterations: int = 30, tag: str = "") -> int:
            current = pattern.copy().astype(float)
            if len(self.etched_memories) > 0:
                norms = [np.linalg.norm(m) + 1e-8 for m in self.etched_memories]
                sims = [np.dot(current.flatten(), m.flatten()) / (np.linalg.norm(current) * n) for m, n in zip(self.etched_memories, norms)]
                avg_sim = np.mean(np.abs(sims)) if sims else 0.0
            else:
                avg_sim = 0.0
            adaptive_lambda = self.fw.decay_lambda_base * (1 - avg_sim)**2

            for _ in range(iterations):
                current = self.fw.cosmo.propagate_vibration(current, distance=1.0, real_freq=freq, custom_lambda=adaptive_lambda)
                current = self.fw.cosmo.perception_fold(current)
            etched = self.fw.utils.compute_equilibrium(current)
            self.etched_memories.append(etched)
            self.tags.append(tag)
            return len(self.etched_memories) - 1

        def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        def recall(self, trigger: np.ndarray, top_k: int = 3) -> List[np.ndarray]:
            if not self.etched_memories:
                return []
            trigger_norm = trigger / (np.linalg.norm(trigger) + 1e-8)
            sims = [self.similarity(trigger_norm, m / (np.linalg.norm(m) + 1e-8)) for m in self.etched_memories]
            indices = np.argsort(sims)[-top_k:][::-1]
            recalled = []
            for idx in indices:
                combined = trigger + self.etched_memories[idx] * (sims[idx] + 1.0)
                recalled.append(self.fw.utils.compute_equilibrium(combined))
            return recalled
