"""
observer.py — Multi-observer communion and brain wave resonance
Ported/adapted from older Pi2 for agentic exploration and consensus etching.
"""

import numpy as np
from typing import List, Optional

brain_wave_bands = {
    'delta': {'mean': 2.25, 'bandwidth': 3.5},
    'theta': {'mean': 6.0, 'bandwidth': 4.0},
    'alpha': {'mean': 10.0, 'bandwidth': 4.0},
    'beta': {'mean': 21.0, 'bandwidth': 18.0},
    'gamma': {'mean': 65.0, 'bandwidth': 70.0}
}

class Observer:
    def __init__(self, framework):
        self.framework = framework

    def blend(self, data: np.ndarray, band: Optional[str] = None, real_freq: Optional[float] = None) -> np.ndarray:
        # Simplified hemisphere blend: equilibrium with optional freq/band boost
        eq = self.framework.utils.compute_equilibrium(data)
        if band:
            band_data = brain_wave_bands.get(band.lower(), {'mean': 1.0})
            boost = band_data['mean'] ** 2 / self.framework.pi_center  # Second-pass squaring
            eq *= (1 + boost)
        elif real_freq:
            eq *= (real_freq ** 2 / self.framework.pi_center)
        return np.clip(eq, *self.framework.base_range)

class MultiObserver:
    def __init__(self, framework, num_observers: int = 3, bands: Optional[List[str]] = None):
        self.framework = framework
        self.observers = [Observer(framework) for _ in range(num_observers)]
        self.bands = bands or ['beta'] * num_observers
        self.cumulative_perturb = np.random.uniform(*framework.base_range)

    def interact(self, data: np.ndarray, iterations: int = 5, refraction_flip: bool = True) -> tuple[float, float]:
        perceptions = [obs.blend(data, band=self.bands[i]) for i, obs in enumerate(self.observers)]
        consensus = np.mean(perceptions, axis=0)

        for _ in range(iterations):
            props = []
            for i, perc in enumerate(perceptions):
                amp = np.mean(perc)
                if refraction_flip and i % 2 == 1:
                    amp = self.framework.cosmo.refract(amp) if hasattr(self.framework, 'cosmo') else -amp
                props.append(self.framework.propagate_vibration(amp) if hasattr(self.framework, 'propagate_vibration') else amp)
            
            linked = self.framework.utils.holographic_linkage(np.array(props)) if hasattr(self.framework.utils, 'holographic_linkage') else np.array(props)
            consensus = self.framework.utils.compute_equilibrium(np.mean(linked))

            # Accumulate for events/communion persistence
            self.cumulative_perturb += np.mean(linked) * self.framework.deviation
            self.cumulative_perturb = np.clip(self.cumulative_perturb, *self.framework.base_range)

        return float(np.mean(consensus)), float(self.cumulative_perturb)