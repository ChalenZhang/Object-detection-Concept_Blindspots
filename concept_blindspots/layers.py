import torch
from torch import nn

class NormalizedTower(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("input_mean", mean.float().reshape(1, input_dim))
        self.register_buffer("input_std", std.float().clamp_min(1e-5).reshape(1, input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs.float() - self.input_mean) / self.input_std
        return self.network(normalized)



class MultiTaskHeads(nn.Module):
    def __init__(self, input_dim: int, group_count: int):
        super().__init__()
        self.binary = nn.Linear(input_dim, group_count)
        self.missed_count = nn.Linear(input_dim, group_count)
        self.q50 = nn.Linear(input_dim, group_count)

    def forward(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "binary_logits": self.binary(representation),
            "log_missed_count": self.missed_count(representation),
            "log_q50": self.q50(representation),
        }

