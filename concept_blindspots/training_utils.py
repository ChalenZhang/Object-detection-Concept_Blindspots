from __future__ import annotations
import math
from dataclasses import dataclass
import torch

class DistinctIndexStream:
    """Yield shuffled fixed-size batches, dropping only the short epoch tail."""

    def __init__(self, count: int, batch_size: int, seed: int):
        if count < batch_size:
            raise SystemExit(
                f"Cannot draw {batch_size} distinct rows from a cohort of {count}"
            )
        self.count = count
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = count

    def draw(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.count:
            self.order = torch.randperm(self.count, generator=self.generator)
            self.cursor = 0
        output = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return output



@dataclass
class SceneViewStream:
    scene_rows: list[torch.Tensor]
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        self.scene_stream = DistinctIndexStream(
            len(self.scene_rows), self.batch_size, self.seed
        )
        self.view_generator = torch.Generator().manual_seed(self.seed + 104729)

    def draw(self) -> torch.Tensor:
        scene_indices = self.scene_stream.draw().tolist()
        rows = []
        for scene_index in scene_indices:
            choices = self.scene_rows[scene_index]
            choice = int(
                torch.randint(
                    len(choices), (1,), generator=self.view_generator
                ).item()
            )
            rows.append(int(choices[choice]))
        return torch.tensor(rows, dtype=torch.long)



def cosine_learning_rate(
    base_lr: float,
    step: int,
    max_steps: int,
    warmup_steps: int = 0,
) -> float:
    if warmup_steps and step <= warmup_steps:
        factor = 0.1 + 0.9 * step / warmup_steps
        return base_lr * factor
    local_step = max(0, step - warmup_steps - 1)
    local_total = max(1, max_steps - warmup_steps)
    progress = min(1.0, local_step / local_total)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))



def missed_object_capture(
    miss_count: torch.Tensor,
    risk: torch.Tensor,
    fraction: float = 0.05,
) -> float:
    total = miss_count.float().sum()
    if total <= 0:
        return float("nan")
    count = max(1, int(math.ceil(fraction * len(risk))))
    order = torch.argsort(risk.float(), descending=True, stable=True)[:count]
    return float(miss_count[order].float().sum() / total)



def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores.float(), stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original = torch.empty_like(ranks)
    original[order] = ranks
    rank_sum = original[labels].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )



def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.float()
    positives = labels.sum()
    if positives <= 0:
        return float("nan")
    order = torch.argsort(scores.float(), descending=True, stable=True)
    local = labels[order]
    precision = local.cumsum(0) / torch.arange(1, len(local) + 1)
    return float((precision * local).sum() / positives)



def calibration_metrics(labels: dict, risk: torch.Tensor) -> dict[str, float]:
    return {
        "capture05": missed_object_capture(labels["miss_count"], risk, 0.05),
        "auroc": binary_auroc(labels["unsafe"], risk),
        "average_precision": average_precision(labels["unsafe"], risk),
    }



def selection_key(metrics: dict[str, float], step: int) -> tuple[float, float, int]:
    capture = metrics["capture05"]
    auroc = metrics["auroc"]
    return (
        capture if math.isfinite(capture) else -math.inf,
        auroc if math.isfinite(auroc) else -math.inf,
        -step,
    )

