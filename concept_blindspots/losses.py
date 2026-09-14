import torch
import torch.nn.functional as F

def streaming_moments(tensor: torch.Tensor, indices: torch.Tensor, block: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(tensor.shape[1], dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for begin in range(0, len(indices), block):
        local = tensor[indices[begin : begin + block]].float()
        total += local.double().sum(dim=0)
        squared += local.double().square().sum(dim=0)
        count += len(local)
    mean = total / max(count, 1)
    variance = (squared / max(count, 1) - mean.square()).clamp_min(1e-10)
    return mean.float(), variance.sqrt().float()



def ranking_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(prediction) < 2:
        return prediction.sum() * 0.0
    paired_prediction = prediction.flip(0)
    paired_target = target.flip(0)
    difference = target - paired_target
    valid = difference != 0
    if not valid.any():
        return prediction.sum() * 0.0
    sign = difference[valid].sign()
    margin = prediction[valid] - paired_prediction[valid]
    return F.softplus(-sign * margin).mean()

