from pathlib import Path
from types import SimpleNamespace
import json

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .layers import NormalizedTower, MultiTaskHeads
from .sae import TopKSAE
from .pooling import summarize_concepts


def load_tensor_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_detector_state(model_dir):
    model_dir = Path(model_dir)
    directory = model_dir / "detector_r101"
    if not directory.is_dir():
        return load_tensor_file(model_dir / "detector_r101.pt")
    parts = sorted(directory.glob("*.pt"))
    if not parts:
        raise FileNotFoundError(f"No detector weights in {directory}")
    state = {}
    for path in parts:
        part = load_tensor_file(path)
        if state.keys() & part.keys():
            raise ValueError(f"Duplicate detector parameters in {path.name}")
        state.update(part)
    return state


class RiskMLP(nn.Module):
    def __init__(self, mean, std, hidden_dim=128, dropout=0.1, groups=5):
        super().__init__()
        self.feature_tower = NormalizedTower(len(mean), mean, std, 2 * hidden_dim, hidden_dim, dropout)
        self.heads = MultiTaskHeads(hidden_dim, groups)

    def forward(self, concept):
        return self.heads(self.feature_tower(concept))

    @classmethod
    def from_checkpoint(cls, checkpoint):
        conf = checkpoint["model_config"]
        model = cls(checkpoint["normalization"]["means"]["concept"], checkpoint["normalization"]["stds"]["concept"], conf["hidden_dim"], conf["dropout"], len(checkpoint["groups"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.eval()


def risk_inputs(concept, config):
    count = len(config["clusters"])
    if concept.ndim != 2 or concept.shape[1] != 4 * count:
        raise ValueError(f"Expected {4 * count} pooled concept channels")
    positions = {row["cluster_id"]: i for i, row in enumerate(config["clusters"])}
    index = torch.tensor([positions[name] for name in config["blindspot_ids"]], device=concept.device)
    selected = torch.cat([concept[:, block * count + index] for block in range(4)], dim=1)
    return torch.cat([concept, selected], dim=1)


@torch.inference_mode()
def predict_risk(concept, model_dir, config, device="cpu", batch_size=256):
    inputs = risk_inputs(concept, config)
    scores = []
    for seed in (2027, 2028, 2029):
        checkpoint = load_tensor_file(Path(model_dir) / f"risk_seed_{seed}.pt")
        model = RiskMLP.from_checkpoint(checkpoint).to(device)
        columns = [checkpoint["groups"].index(name) for name in ("car", "person")]
        parts = []
        for begin in range(0, len(inputs), batch_size):
            output = model(inputs[begin:begin + batch_size].to(device))
            counts = torch.expm1(output["log_missed_count"].clamp(-8, 8)).clamp_min(0)
            parts.append(counts[:, columns].sum(dim=1).cpu().double())
        scores.append(torch.cat(parts))
    return ((scores[0] + scores[1] + scores[2]) / len(scores)).float()


class ConceptBank:
    def __init__(self, path, config, device):
        saved = load_tensor_file(path)
        self.families = [SimpleNamespace(**row) for row in config["clusters"]]
        self.risk = torch.tensor([row.excess_risk for row in self.families])
        self.blindspot = torch.tensor([float(row.blindspot) for row in self.families])
        self.models, self.means, self.stds = {}, {}, {}
        self.class_families, self.global_offsets = {}, {}
        for key, payload in saved.items():
            class_id = int(key)
            c = payload["model_config"]
            model = TopKSAE(c["dim"], c["num_concepts"], c["topk"])
            model.load_state_dict(payload["state_dict"], strict=True)
            self.models[class_id] = model.to(device).eval()
            self.means[class_id] = payload["mean"].to(device).float()
            self.stds[class_id] = payload["std"].to(device).float().clamp_min(1e-6)
            indices = [i for i, row in enumerate(self.families) if row.class_id == class_id]
            self.class_families[class_id] = [self.families[i] for i in indices]
            self.global_offsets[class_id] = (indices[0], indices[-1] + 1)

    @torch.no_grad()
    def encode(self, features, class_id):
        normalized = (features.float() - self.means[class_id]) / self.stds[class_id]
        code = self.models[class_id].encode(normalized)
        values = [code[:, row.members].sum(dim=1) / row.scale for row in self.class_families[class_id]]
        return torch.stack(values, dim=1).clamp_min(0)


class ImageRiskModel:
    def __init__(self, model_dir="models", config_dir=None, device="cuda", amp=True):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[1] / "configs"
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.model_dir = Path(model_dir)
        self.config = json.loads((Path(config_dir) / "concepts.json").read_text())
        settings = json.loads((Path(config_dir) / "detector.json").read_text())
        self.args = SimpleNamespace(**settings, concept_class_ids=(3, 5, 6))
        backbone = resnet_fpn_backbone(backbone_name="resnet101", weights=None, trainable_layers=3)
        self.detector = FasterRCNN(backbone, num_classes=len(settings["classes"]), min_size=settings["min_size"], max_size=settings["max_size"])
        self.detector.load_state_dict(load_detector_state(self.model_dir), strict=True)
        self.detector.roi_heads.score_thresh = settings["score_threshold"]
        self.detector.roi_heads.detections_per_img = settings["detections_per_image"]
        self.detector.to(self.device).eval()
        self.bank = ConceptBank(self.model_dir / "detsae.pt", self.config, self.device)

    @torch.inference_mode()
    def extract(self, images):
        images = [image.to(self.device) for image in images]
        sizes = [tuple(image.shape[-2:]) for image in images]
        model = self.detector
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            image_list, _ = model.transform(images, None)
            body = model.backbone.body(image_list.tensors)
            fpn = model.backbone.fpn(body)
            proposals, _ = model.rpn(image_list, fpn, None)
            pooled = model.roi_heads.box_roi_pool(fpn, proposals, image_list.image_sizes)
            features = model.roi_heads.box_head(pooled)
            logits, regression = model.roi_heads.box_predictor(features)
        probabilities = logits.float().softmax(dim=1)
        offsets = [0]
        for proposal in proposals:
            offsets.append(offsets[-1] + len(proposal))
        concept, _ = summarize_concepts(self.bank, features, proposals, probabilities, offsets, self.args)
        boxes, scores, labels = model.roi_heads.postprocess_detections(logits.float(), regression.float(), proposals, image_list.image_sizes)
        detections = [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(boxes, scores, labels)]
        detections = model.transform.postprocess(detections, image_list.image_sizes, sizes)
        return concept.cpu().half(), [{k:v.cpu() for k,v in row.items()} for row in detections]
