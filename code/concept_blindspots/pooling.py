from __future__ import annotations
import argparse
import torch
from torchvision.ops import nms

def spatial_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    candidates: int,
    keep: int,
    iou_threshold: float,
) -> torch.Tensor:
    if not len(scores):
        return torch.empty(0, dtype=torch.long, device=scores.device)
    count = min(candidates, len(scores))
    top = torch.topk(scores, count).indices
    return top[nms(boxes[top].float(), scores[top].float(), iou_threshold)[:keep]]



def summarize_concepts(
    bank: FrozenConceptBank,
    features: torch.Tensor,
    boxes: list[torch.Tensor],
    probabilities: torch.Tensor,
    offsets: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    family_count = len(bank.families)
    concept_rows = []
    prior_rows = []
    risk = bank.risk.to(features.device)
    blindspot = bank.blindspot.to(features.device)
    for image_index, local_boxes in enumerate(boxes):
        begin, end = offsets[image_index], offsets[image_index + 1]
        maxima = features.new_zeros(family_count).float()
        top3 = features.new_zeros(family_count).float()
        weighted_maxima = features.new_zeros(family_count).float()
        weighted_top3 = features.new_zeros(family_count).float()
        for class_id in args.concept_class_ids:
            class_probability = probabilities[begin:end, class_id]
            keep = spatial_indices(
                local_boxes,
                class_probability,
                args.proposals_per_class,
                args.spatial_proposals_per_class,
                args.proposal_nms_threshold,
            )
            if not len(keep):
                continue
            strength = bank.encode(features[begin:end][keep], class_id)
            weighted = strength * class_probability[keep, None].float()
            local_k = min(3, len(keep))
            start, stop = bank.global_offsets[class_id]
            maxima[start:stop] = strength.max(dim=0).values
            top3[start:stop] = strength.topk(local_k, dim=0).values.mean(dim=0)
            weighted_maxima[start:stop] = weighted.max(dim=0).values
            weighted_top3[start:stop] = weighted.topk(local_k, dim=0).values.mean(dim=0)
        concept_rows.append(
            torch.cat([maxima, top3, weighted_maxima, weighted_top3])
        )
        prior_rows.append(
            torch.stack(
                [
                    torch.dot(weighted_maxima, risk),
                    torch.dot(weighted_top3, risk),
                    torch.dot(weighted_maxima, blindspot),
                    torch.dot(weighted_top3, blindspot),
                ]
            )
        )
    return torch.stack(concept_rows), torch.stack(prior_rows)

