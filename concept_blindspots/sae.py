from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def topk_relu(preactivations: torch.Tensor, topk: int) -> torch.Tensor:
    z = F.relu(preactivations)
    if 0 < int(topk) < z.shape[1]:
        values, indices = torch.topk(z, int(topk), dim=1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(1, indices, values)
        return sparse
    return z



class TopKSAE(nn.Module):
    """Top-K SAE with unit-norm decoder atoms to remove scale ambiguity."""

    model_type = "topk_sae"

    def __init__(self, dim: int, num_concepts: int, topk: int):
        super().__init__()
        self.dim = int(dim)
        self.num_concepts = int(num_concepts)
        self.topk = int(topk)
        self.encoder = nn.Linear(self.dim, self.num_concepts)
        self.decoder_atoms = nn.Parameter(torch.empty(self.num_concepts, self.dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder_atoms, std=1.0 / math.sqrt(self.dim))
        self.normalize_decoder_()

    @torch.no_grad()
    def initialize_from_samples(self, samples: torch.Tensor, seed: int) -> None:
        generator = torch.Generator(device=samples.device).manual_seed(int(seed))
        selected = torch.randperm(samples.shape[0], generator=generator, device=samples.device)
        selected = selected[: self.num_concepts]
        atoms = F.normalize(samples[selected], dim=1)
        self.decoder_atoms.copy_(atoms)
        self.encoder.weight.copy_(atoms)
        self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder_atoms.norm(dim=1).clamp_min(1e-8)
        self.decoder_atoms.div_(norms[:, None])
        # Compensate the encoder so normalization does not arbitrarily shrink codes.
        self.encoder.weight.mul_(norms[:, None])
        self.encoder.bias.mul_(norms)

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_relu(self.pre_activations(x), self.topk)

    def atoms(self) -> torch.Tensor:
        return self.decoder_atoms

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.atoms()

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decode(z)

    def anchor_loss(self) -> torch.Tensor:
        return self.decoder_atoms.new_zeros(())

    def state_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "dim": self.dim,
            "num_concepts": self.num_concepts,
            "topk": self.topk,
        }

