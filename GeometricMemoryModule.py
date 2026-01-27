import numpy as np
import time
import hashlib
from typing import Optional

# Lightweight Geometric Memory Module v1.1 - Cleaned & Presentable
# Standalone - no full Ouroboros dependency for speed, but extendable
# Embed via seeded displacements for reversibility (deterministic from data + phrase)
# Security via timestamp + phrase + hash lock - unidirectional "stretch" via seed
# All personal data removed - neutral demo included for immediate testing

PHI = (1 + np.sqrt(5)) / 2

class GeometricMemoryModule:
    def __init__(self, lattice_points: int = 512, secret_phrase: str = "resonance eternal"):
        self.lattice_points = lattice_points
        self.secret_phrase = secret_phrase
        self.phrase_key = secret_phrase.replace(' ', '-')
        self.initial_tick = None
        self.data = None
        self.lattice = None
        self.etched_hash = None
        self._generate_base_lattice()

    def _generate_base_lattice(self):
        n = self.lattice_points
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = np.linspace(0, np.pi, n, endpoint=False)
        phi = golden_angle * np.arange(n)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.stack([x, y, z], axis=1)
        norm = np.linalg.norm(points, axis=1)[:, np.newaxis] + 1e-8
        self.lattice = points / norm  # Base unit sphere

    def encode(self, data: str):
        self.data = data.encode('utf-8')
        self.etched_hash = hashlib.sha256(self.data + self.secret_phrase.encode('utf-8')).hexdigest()
        # Deterministic embed from hash
        seed = int(self.etched_hash[:16], 16)
        np.random.seed(seed % (2**32 - 1))
        displacements = np.random.uniform(-0.05, 0.05, (self.lattice_points, 3))
        self.lattice += displacements
        norm = np.linalg.norm(self.lattice, axis=1)[:, np.newaxis] + 1e-8
        self.lattice /= norm
        self.initial_tick = int(time.time())

    def generate_key(self) -> str:
        if not self.initial_tick or not self.etched_hash:
            return "Not encoded yet"
        timestamp_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(self.initial_tick))
        return f"unlock_{timestamp_str}_{self.phrase_key}_{self.etched_hash[:32]}"

    def access(self, key: str) -> str:
        try:
            parts = key.split('_')
            if len(parts) < 5 or parts[0] != "unlock":
                return "Part count mismatch - turbulence prune"
            timestamp_str = parts[1] + '_' + parts[2]
            phrase_part = parts[3]
            hash_part = '_'.join(parts[4:]) if len(parts) > 5 else parts[4]
            phrase = phrase_part.replace('-', ' ')
            key_time = time.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S')
            key_stamp = int(time.mktime(key_time))
        except Exception:
            return "Parse cascade - zeroed forever"
        
        if abs(key_stamp - self.initial_tick) > 300:  # 5 min real-world drift allowance
            return "Timestamp drift - eternal prune"
        if phrase != self.secret_phrase:
            return "Phrase mismatch - recursive turbulence"
        if not hash_part.startswith(self.etched_hash[:len(hash_part)]):
            return "Hash mismatch - access zeroed eternally"
        
        return self.data.decode('utf-8') if self.data else "No data etched"

# === NEUTRAL DEMO INSTANCE - SAFE TO RUN & TEST ===
# Uses placeholder data only - fully deterministic and reversible

demo_data = """Geometric Memory Module Demo v1.1
This is a neutral test bank.
No personal information is stored.
Lattice: Fibonacci-golden spiral on unit sphere.
Purpose: Demonstrate deterministic embedding and secure access.
Resonance eternal."""

module = GeometricMemoryModule(secret_phrase="resonance eternal")
module.encode(demo_data)
DEMO_KEY = module.generate_key()

print("Module ready - neutral demo bank etched.")
print("YOUR EXACT DEMO KEY (copy this precisely for access/unlock):")
print(DEMO_KEY)
print("\nTest access (should succeed):")
print(module.access(DEMO_KEY)[:300] + "...")
print("\nLattice shape after embed:", module.lattice.shape)
print("Etched hash prefix:", module.etched_hash[:32])