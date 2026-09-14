import torch
from torchvision.ops import box_iou

def match_class(
    gt_boxes: list[list[float]],
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, float, float]:
    gt = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
    matched = torch.zeros(len(gt), dtype=torch.bool)
    quality = torch.zeros(len(gt), dtype=torch.float32)
    localization_loss = 0.0
    true_positives = 0
    if len(gt) and len(pred_boxes):
        overlaps = box_iou(pred_boxes.float(), gt)
        for prediction_index in torch.argsort(pred_scores, descending=True).tolist():
            available = overlaps[prediction_index].clone()
            available[matched] = -1.0
            best_iou, best_gt = available.max(dim=0)
            if float(best_iou) < iou_threshold:
                continue
            matched[best_gt] = True
            quality[best_gt] = pred_scores[prediction_index] * best_iou
            localization_loss += 1.0 - float(best_iou)
            true_positives += 1
    return (
        true_positives,
        len(pred_boxes) - true_positives,
        float((1.0 - quality).sum()),
        localization_loss,
    )



def match_group(
    annotations: list[dict],
    prediction: dict,
    class_ids: tuple[int, ...],
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> dict[str, float]:
    labels = prediction["labels"].long()
    scores = prediction["scores"].float()
    boxes = prediction["boxes"].float()
    gt_count = 0
    true_positives = 0
    false_positives = 0
    q50_loss = 0.0
    localization_loss = 0.0
    for class_id in class_ids:
        local_gt = [row["box"] for row in annotations if row["class_id"] == class_id]
        keep = torch.nonzero(
            (labels == class_id) & (scores >= score_threshold), as_tuple=False
        ).flatten()
        if len(keep):
            keep = keep[torch.argsort(scores[keep], descending=True)[:max_detections]]
        local_tp, local_fp, local_q50, local_localization = match_class(
            local_gt, boxes[keep], scores[keep], iou_threshold
        )
        gt_count += len(local_gt)
        true_positives += local_tp
        false_positives += local_fp
        q50_loss += local_q50
        localization_loss += local_localization
    misses = gt_count - true_positives
    return {
        "gt_count": float(gt_count),
        "has_miss": float(misses > 0),
        "miss_count": float(misses),
        "false_positive_count": float(false_positives),
        "q50_loss": float(q50_loss),
        "localization_loss": float(localization_loss),
        "correction_burden": float(misses + false_positives) + localization_loss,
    }

