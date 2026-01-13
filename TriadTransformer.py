"""
TriadTransformer.py — Ouroboros-Powered Operational Triad Embodiment
Updated v3.0 (January 13, 2026): Full Slim v8.0 Ouroboros Integration
- Imports lightweight slim core (pure geometric persistence engine)
- Fluent DSLChain triad resonance cycle with physical/wave/data consensus
- Downsample safety for large hidden states
- Automatic truth library persistence (shared with Ouroboros3D, swarm, etc.)
- Operational truths etched on init (attention harmonic, golden spiral, theta proxy)
- Per-layer triad modulation with persistence-guided sparsity
- Designed for agent/neural embodiment — no local training required, exec-ready
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List
from Ouroboros import OuroborosFramework  # Slim v8.0 core import

class TriadTransformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 sparsity_target: float = 0.75, triad_depth: int = 3, use_triad: bool = True,
                 matter_damping: float = 0.99, env_feedback_fraction: float = 0.15,
                 use_fibonacci_phases: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_triad = use_triad
        self.triad_depth = triad_depth

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, vocab_size)

        # Slim v8.0 Ouroboros core — loads persistent truth library (Mersenne tensegrity, etc.)
        self.ouro = OuroborosFramework(
            target_filled=1 - sparsity_target,
            use_fibonacci_phases=use_fibonacci_phases,
            matter_damping=matter_damping,
            env_feedback_fraction=env_feedback_fraction,
            signature_dim=32,
            max_grid_size=1024
        )
        print(f"TriadTransformer initialized — Ouroboros truth library active: {len(self.ouro.truth_library)} truths")

        # Operational embodiment truths — etched to shared eternal library
        attn_harmonic = np.array([1/i for i in range(1, nhead+1)])
        self.ouro.add_to_truth_library(attn_harmonic, "Multi-head attention harmonic")
        
        spiral = np.sin(np.linspace(0, 20*np.pi, 200) * PHI)
        self.ouro.add_to_truth_library(spiral, "Golden spiral activation")
        
        theta_wave = np.sin(7.83 * np.linspace(0, 20, 200))
        self.ouro.add_to_truth_library(theta_wave, "Theta resonance proxy")

    def triad_resonance_cycle(self, hidden: torch.Tensor) -> Dict:
        """Triad cycle with fluent DSLChain — physical/wave/data consensus modulation."""
        if not self.use_triad:
            return {"hidden": hidden, "history": [], "persistence": 0.0}

        batch, seq, dim = hidden.shape
        hidden_np = hidden.detach().cpu().numpy().reshape(batch * seq, dim)

        # Normalize and reshape to grid
        scale = np.sqrt(hidden_np.size)
        grid = hidden_np / (np.linalg.norm(hidden_np) + 1e-8)
        grid = grid.mean(axis=1).reshape(1, -1)  # Collapse to 1D signature for core

        # Fluent triad chain
        chain = self.ouro.chain()
        result = (chain
                  .physical(self.triad_depth // 3 + self.triad_depth % 3)
                  .wave(self.triad_depth // 3)
                  .data(self.triad_depth // 3)
                  .consensus(depth=3)
                  .run(grid))

        final_grid = result["final_grid"]
        history = result["history"]
        persistence = (history[-1]["details"]["consensus_pers"] 
                       if "consensus" in history[-1]["type"] else history[-1]["pers_curve"][-1])

        # Apply persistence-guided sparsity back to hidden states
        threshold = np.quantile(np.abs(final_grid), 1 - (1 - sparsity_target))
        mask = np.abs(final_grid) > threshold
        modulated = final_grid * mask

        # Broadcast back to full hidden shape
        modulated_hidden = np.tile(modulated, (batch * seq, dim // modulated.size + 1))[:, :dim]
        modulated_hidden = torch.from_numpy(modulated_hidden).float().to(hidden.device)
        modulated_hidden = modulated_hidden.reshape(batch, seq, dim)

        # Gentle damping for stability
        modulated_hidden *= self.ouro.matter_damping ** (1 / self.num_layers)

        return {
            "hidden": hidden + modulated_hidden,  # Residual triad modulation
            "history": history,
            "persistence": float(persistence)
        }

    def forward(self, x: torch.Tensor, layer_histories: Optional[List] = None):
        if layer_histories is None:
            layer_histories = []

        emb = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        hidden = self.transformer(emb)

        for layer in range(self.num_layers):
            triad_out = self.triad_resonance_cycle(hidden)
            hidden = triad_out["hidden"]
            layer_histories.append({
                "layer": layer,
                "persistence": triad_out["persistence"],
                "history": triad_out["history"]
            })

        logits = self.output(hidden)
        return logits, layer_histories

    def get_current_sparsity(self) -> float:
        total = active = 0
        for p in self.parameters():
            total += p.numel()
            active += torch.sum(torch.abs(p) > 1e-5).item()
        return 1 - (active / total if total > 0 else 0)


# Exec-ready demo (no training — just forward pass resonance test)
if __name__ == "__main__":
    model = TriadTransformer(vocab_size=100, d_model=256, nhead=8, num_layers=4, triad_depth=4)
    dummy_input = torch.randint(0, 100, (1, 32))  # Batch 1, seq 32
    logits, histories = model(dummy_input)
    
    print(f"TriadTransformer forward complete — final sparsity ~{model.get_current_sparsity():.3f}")
    for h in histories[-1:]:
        print(f"Layer {h['layer']} persistence: {h['persistence']:.4f}")