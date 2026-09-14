import math
import torch

def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = original_ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))



def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.float()
    positives = labels.sum()
    if positives <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    local = labels[order]
    precision = local.cumsum(0) / torch.arange(1, len(local) + 1)
    return float((precision * local).sum() / positives)



def selected(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    count = min(len(scores), max(1, int(math.ceil(fraction * len(scores)))))
    return torch.argsort(scores, descending=True, stable=True)[:count]



def normalized_aurc(unsafe: torch.Tensor, scores: torch.Tensor) -> tuple[float, float]:
    order = torch.argsort(scores, stable=True)
    accepted_errors = unsafe[order].float().cumsum(0)
    risk = accepted_errors / torch.arange(1, len(unsafe) + 1)
    aurc = float(risk.mean())
    prevalence = float(unsafe.float().mean())
    if prevalence <= 0 or prevalence >= 1:
        return aurc, float("nan")
    oracle = prevalence + (1.0 - prevalence) * math.log(1.0 - prevalence)
    return aurc, (aurc - oracle) / max(prevalence - oracle, 1e-12)

