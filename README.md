# Ouroboros Framework

A scale-invariant geometric framework for exploring persistence, resonance, and cross-scale patterns.

## Overview

Ouroboros models a spherical manifold where π transitions from ≈3.1416 (flat center) to a curved boundary (default edge ≈2.094). A fixed deviation=2 creates natural thirds divisions and dual-pass dynamics: bloom (expansion with stochastic kick) followed by etching (squaring amplification + irreversible pruning).

The framework derives parameter-free patterns that echo real-world phenomena across scales—from cosmology to biology.

**Key Features**:
- Parameter-free cosmic density approximations (with time-loss evolution).
- Sub-space scans on arbitrary data (images, sequences, networks).
- EM-inspired pulses (matter/data interaction).
- Plant manifold simulations (light/sound rebound for growth/resilience).

## New in v2: EM Matter/Data Interaction

New Method: 
Adds the separation/transmission insight (data bridge across voids, high vibration proxy).
```
simulate_manifold_slice_pull
visualize_slice_pull—directly
```
Recent insights integrate electromagnetism:
- **Photons** as massless "data couriers" (fast, eternal propagation trails).
- **Electrons** as massive "etchers" (negative charge probing positive moat, repulsion preventing collapse).
- Matter persists via push/pull balance; data flows as EM waves modulated by charge geometry.

This closes the matter-data loop—Ouroboros captures EM contrast in persistence dynamics.

## Number Theory Probes
Probe for perfect number patterns—even as pressure points (high persistence symmetry), guiding odds to auto-prune (asymmetry repulsion via pi differential).

Quick start:
```
Pythoneven, odd = ouro.probe_perfect_numbers(exponent_range=10, odd_check=True)
print("Even perfect pressure points:", even)
print("Odd prune candidates:", odd)
```
Ouroboros Demo:
```
Base densities: (0.3436592257647936, 0.6563407742352064)
Observed (time-loss): (0.29623425260925207, 0.7037657473907479)

Sub-space scan persistence: 0.6656

EM pulse (red bloom): Persistence 0.8800, Reclaimed 8.6000

Plant pulse (220Hz rebound): Persistence 0.7600, Reclaimed 35.2700

Even perfect pressure points: [6, 28, 496, 8128]
Odd prune candidates: ['Odd candidate pruned at p=2', 'Odd candidate pruned at p=3',
'Odd candidate pruned at p=4', 'Odd candidate pruned at p=5', 'Odd candidate pruned at p=6',
'Odd candidate pruned at p=7', 'Odd candidate pruned at p=8', 'Odd candidate pruned at p=9',
'Odd candidate pruned at p=10']
```
## Sub-Space Scans

Treat any grid/data as manifold layers—propagate dual-pass to reveal persistent patterns.

Quick start:
```python
from Ouroboros import OuroborosFramework
import numpy as np

ouro = OuroborosFramework()
grid = np.random.uniform(-1, 1, (50, 50))  # Or load image/sequence
scanned, persistence = ouro.subspace_scan(grid)
print(f"Persistence: {persistence:.4f}")
# Visualize scanned heatmap with matplotlib
Plant Pulse Simulations
Pulse plant manifolds with light/sound frequencies—rebound via thirds asymmetry for amplification.
```
Quick start:
```
Pythonpers, reclaimed, wave = ouro.pulse_plant_manifold(base_freq=220.0, rebound_amp=True)
print(f"Growth proxy: {pers:.4f}, Reclaimed stress energy: {reclaimed:.4f}")
```
Red light proxy (660nm) for bloom/yield; low Hz sound (220Hz) for resilience rebound.

## Images

Images were created from the new sub-space pulse feature via the following simple changes to the end of the code(visuals section).

Save visualizations as PNGs for repo:
```
    ouro.visualize_time_flow(save_path="time_flow_trails.png")
    ouro.visualize_ring_manifold_time_flow(save_path="ring_manifold_trails.png")
    ouro.visualize_manifold(save_path="3d_manifold.png")
```

## Visualizations
Run the script for interactive plots:
- Time Flow Trails - Low persistence = fleeting ghosts; high = enduring sharp paths.
- Ring Manifold - Inner high persistence = dense filaments; outer = fading voids.
- 3D Manifold - Colored by π gradient with thirds division planes.

## Applications:
- Pattern Discovery: Sub-space scans on images/DNA/networks reveal hidden persistence.
- Biology Modeling: Simulate light/sound pulses on plant manifolds for growth/resilience insights.
- Resilience Simulations: Test tamper/containment (inspiration for EtchSecure tools).
- Educational/Research: Explore scale-invariant geometry and EM matter/data interplay.

## Disclaimer
This is an experimental/research tool—speculative unification of patterns across scales. Not production software or validated science. Use for exploration and fun.

Installation:
```
Bashgit clone https://github.com/wwes4/Ouroboros.git
cd Ouroboros
pip install -r requirements.txt
requirements.txt:
textnumpy>=1.21.0
matplotlib>=3.5.0
sympy>=1.9
```
Quick Start:
```
Pythonpython Ouroboros.py
```
Explore methods interactively—tweak parameters and re-run.
MIT License – free to use, extend, and share.
⭐ Star if the patterns resonate!
Issues/ideas welcome.
By @SquaredGradient | January 2026
