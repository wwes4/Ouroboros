"""
vibration.py — Resilient propagation and linkage
Ported essentials for waveform-to-resonance flow.
"""

import numpy as np

def propagate_vibration(framework, amp: float, dist: float = 10.0, position_ratio: float = 0.5) -> float:
    decayed = amp * np.exp(-0.01 * dist * (framework.deviation / framework.pi_center))
    tension = framework.hybrid_tension(position_ratio) if hasattr(framework, 'hybrid_tension') else 1.0
    scaled = decayed * tension
    scaled = np.clip(scaled, *framework.base_range)
    return framework.utils.compute_equilibrium(np.array([scaled]))[0]

def refract(framework, amp: float, position_ratio: float = 0.5) -> float:
    tilde_c = framework.c_base * (1 - framework.deviation * position_ratio ** framework.scale_factor)
    n = framework.c_base / max(tilde_c, framework.min_c)
    bent = amp * (n - 1) * framework.deviation
    return np.clip(bent, *framework.base_range)

def holographic_linkage(framework, data: np.ndarray, position_ratio: float = 0.5, real_freq: Optional[float] = None) -> np.ndarray:
    if real_freq:
        freq = real_freq / framework.pi_center
    else:
        freq = 1.0
    chain = np.fft.fft(data)
    realized = np.real(chain * freq ** 2)  # Second-pass squaring
    return framework.utils.compute_equilibrium(realized)