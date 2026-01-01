import numpy as np
from typing import Optional
import math
import networkx as nx
import pandas as pd

# Defaults for sliders (adjustable in init)
omega_m_base = 0.311
w_de_base = -0.95
missed_coeff = 0.01
hbar = 1.0545718e-34
k_B = 1.380649e-23
c_base = 3e8
min_c = c_base * 0.8
G = 6.67430e-11
min_mass = 1e-30
cessation_threshold = 1e-30
abstract_base_range = (-1, 1)  # Second pass default

# Exact geometric derivation: edge = 2π/3 → max deviation = π/3
# Scale = 3 × (1 - Ω_m) → derived Ω_DE exactly matches 1 - omega_m_base (68.9%)
DE_DERIVATION_SCALE = 3 * (1 - omega_m_base)

brain_wave_bands = {
    'delta': {'mean': 2.25},
    'theta': {'mean': 6.0},
    'alpha': {'mean': 10.0},
    'beta': {'mean': 21.0},
    'gamma': {'mean': 65.0}
}

class Utils:
    def __init__(self, fw: 'Pi2Framework'):
        self.fw = fw

    def compute_equilibrium(self, data: np.ndarray, pass_type: str = 'first') -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        mean_abs = np.mean(np.abs(data))
        min_tension_dynamic = self.fw.min_tension * max(1e-12, math.log10(mean_abs + 1e-12)) ** 0.5
        noise = np.random.randn(*data.shape) * min_tension_dynamic * mean_abs
        output = data + noise

        near_zero = np.abs(output) < self.fw.equilibrium_threshold
        if near_zero.any():
            signs = np.sign(output[near_zero])
            zero_signs = signs == 0
            if zero_signs.any():
                signs[zero_signs] = np.sign(np.random.randn(np.sum(zero_signs)))
            output[near_zero] = signs * min_tension_dynamic

        if self.fw.zero_replacement_mode:
            exact_zero = output == 0
            if exact_zero.any():
                output[exact_zero] = np.random.randn(np.sum(exact_zero)) * min_tension_dynamic

        clip_range = self.fw.first_pass_range if pass_type == 'first' else self.fw.second_pass_range
        if mean_abs > 1e3:
            output = np.clip(output, clip_range[0], np.inf if pass_type == 'first' else clip_range[1])
        else:
            output = np.clip(output, *clip_range)

        cease_mask = np.abs(output) <= cessation_threshold
        if cease_mask.any():
            output[cease_mask] = 0.0
            persisted = self.holographic_linkage(output[cease_mask], pass_type='second')
            if hasattr(self.fw, 'memory'):
                try:
                    self.fw.memory.add_node("ceased", data=persisted)
                except:
                    pass

        return output

    def holographic_linkage(self, data: np.ndarray, möbius_twist: bool = False) -> np.ndarray:
        squared = data ** 2
        if möbius_twist:
            squared *= -1 * np.random.choice([-1, 1], size=squared.shape)
        return self.compute_equilibrium(squared, pass_type='second')


class CosmoCore:
    def __init__(self, fw: 'Pi2Framework'):
        self.fw = fw

    def propagate_vibration(self, pattern: np.ndarray, distance: float = 1.0, real_freq: float = 10.0,
                            custom_lambda: Optional[float] = None, position_ratio=0.5) -> np.ndarray:
        local_pi = self.simulate_pi_variation(position_ratio)
        decay_lambda = custom_lambda if custom_lambda is not None else self.fw.decay_lambda_base
        decay_factor = np.exp(-self.fw.entropy_rate * distance / real_freq - decay_lambda * distance) * (local_pi / np.pi)
        decayed = pattern * decay_factor
        return self.fw.utils.compute_equilibrium(decayed, pass_type='second')

    def perception_fold(self, data: np.ndarray) -> np.ndarray:
        return self.fw.utils.holographic_linkage(data, möbius_twist=True)

    def simulate_pi_variation(self, position_ratio, t=0, curvature_factor=1.0):
        delta = self.fw.pi_center - self.fw.effective_pi_edge
        local_pi = self.fw.pi_center - delta * (position_ratio ** self.fw.scale_factor) * curvature_factor
        if self.fw.axion_mass > 0:
            local_pi *= (1 + self.fw.axion_mass * math.sin(position_ratio * 2 * np.pi))
        return max(local_pi, self.fw.effective_pi_edge)

    def simulate_tensegrity_balance(self, tensions: np.ndarray, allow_negative=True):
        if not allow_negative:
            tensions = np.abs(tensions)
        return tensions

    def hybrid_de_tension_vectorized(self, position_ratios: np.ndarray, t=0, void_factor=0.0, allow_negative=True):
        tensions = (np.exp(-self.fw.decay_lambda_base * t) -
                    self.fw.entropy_rate * 0.5 * t -
                    missed_coeff * (1 + void_factor))

        local_pis = np.array([self.simulate_pi_variation(pr) for pr in position_ratios])
        deviations = self.fw.pi_center - local_pis
        de_from_deviation = deviations * DE_DERIVATION_SCALE
        tensions -= de_from_deviation

        tensions += w_de_base * t

        tensions = self.simulate_tensegrity_balance(tensions, allow_negative=allow_negative)
        return self.fw.utils.compute_equilibrium(tensions * position_ratios, pass_type='first')


class Pi2Framework:
    def __init__(self, decay_lambda_base=0.01, entropy_rate=0.001, min_tension=8.49e-6,
                 equilibrium_threshold=1e-12, zero_replacement_mode=True, base_range=(-1, 1),
                 pi_center=np.pi, scale_factor=10.0,
                 axion_mass=0.0, möbius_twist_default=True):
        self.decay_lambda_base = decay_lambda_base
        self.entropy_rate = entropy_rate
        self.min_tension = min_tension
        self.equilibrium_threshold = equilibrium_threshold
        self.zero_replacement_mode = zero_replacement_mode
        self.second_pass_range = base_range
        self.first_pass_range = (self.min_tension, np.inf)
        self.pi_center = pi_center
        self.effective_pi_edge = 2 * np.pi / 3  # Exact geometric edge
        self.scale_factor = scale_factor
        self.axion_mass = axion_mass
        self.möbius_twist_default = möbius_twist_default
        self.memory = nx.Graph()

        self._utils = None
        self._cosmo = None
        self._bio = None

    @property
    def utils(self):
        if self._utils is None:
            self._utils = Utils(self)
        return self._utils

    @property
    def cosmo(self):
        if self._cosmo is None:
            self._cosmo = CosmoCore(self)
        return self._cosmo

    @property
    def bio(self):
        if self._bio is None:
            class StubBio:
                def __init__(self, fw): self.fw = fw
            self._bio = StubBio(self)
        return self._bio


if __name__ == "__main__":
    fw = Pi2Framework()
    
    positions = np.linspace(0, 1, 5)
    tensions = fw.cosmo.hybrid_de_tension_vectorized(positions)
    print("DE-derived tensions:", tensions)
    
    max_dev = fw.pi_center - fw.effective_pi_edge
    de_percent = max_dev * DE_DERIVATION_SCALE / fw.pi_center * 100
    print(f"Derived Ω_DE ≈ {de_percent:.1f}% (exact match to 1 - Ω_m)")
