import torch
from torchvision.ops import box_iou


def ignored_kitti_predictions(annotations, prediction, original,
                              threshold=0.5, iou_threshold=0.5, cap=100):
    ignored, raw_fp = 0, 0
    for class_id, neighbor in ((3, "Van"), (5, "Person_sitting")):
        keep = torch.nonzero((prediction["labels"] == class_id) &
                             (prediction["scores"] >= threshold)).flatten()
        keep = keep[torch.argsort(prediction["scores"][keep], descending=True)[:cap]]
        boxes = prediction["boxes"][keep].float()
        scores = prediction["scores"][keep]
        gt = torch.tensor([row["box"] for row in annotations if row["class_id"] == class_id]).float().reshape(-1, 4)
        unmatched = torch.ones(len(boxes), dtype=torch.bool)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        if len(gt) and len(boxes):
            overlaps = box_iou(boxes, gt)
            for index in torch.argsort(scores, descending=True).tolist():
                available = overlaps[index].clone()
                available[matched] = -1
                best, target = available.max(0)
                if float(best) >= iou_threshold:
                    matched[target] = True
                    unmatched[index] = False
        boxes = boxes[unmatched]
        raw_fp += len(boxes)
        neighbors = torch.tensor([row["bbox_xyxy"] for row in original if row["type"] == neighbor]).float().reshape(-1, 4)
        excluded = torch.zeros(len(boxes), dtype=torch.bool)
        if len(neighbors) and len(boxes):
            overlaps = box_iou(boxes, neighbors)
            matched_neighbors = torch.zeros(len(neighbors), dtype=torch.bool)
            for index in range(len(boxes)):
                available = overlaps[index].clone()
                available[matched_neighbors] = -1
                best, target = available.max(0)
                if float(best) >= iou_threshold:
                    excluded[index] = True
                    matched_neighbors[target] = True
        regions = torch.tensor([row["bbox_xyxy"] for row in original if row["type"] == "DontCare"]).float().reshape(-1, 4)
        if len(regions) and len(boxes):
            lower = torch.maximum(boxes[:, None, :2], regions[None, :, :2])
            upper = torch.minimum(boxes[:, None, 2:], regions[None, :, 2:])
            intersection = (upper - lower).clamp_min(0).prod(2)
            area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1).clamp_min(1e-8)
            excluded |= (intersection / area[:, None]).max(1).values >= iou_threshold
        ignored += int(excluded.sum())
    return ignored, raw_fp


def error_counts(annotations, prediction, class_ids=(3, 5), task="fn", kitti=None):
    if task not in ("fn", "fp"):
        raise ValueError("Task must be fn or fp")
    threshold = 0.5 if task == "fp" else 0.05
    result = match_group(annotations, prediction, class_ids, threshold, 0.5, 100)
    if task == "fn":
        return result["miss_count"]
    count = result["false_positive_count"]
    if kitti is not None:
        local = dict(prediction)
        local["labels"] = prediction["labels"].clone()
        include = torch.zeros_like(local["labels"], dtype=torch.bool)
        for class_id in class_ids:
            include |= local["labels"] == class_id
        local["labels"][~include] = -1
        ignored, _ = ignored_kitti_predictions(annotations, local, kitti)
        count -= ignored
    return count

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
