"""Rank local images by predicted missed-object risk."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concept_blindspots.model import ImageRiskModel, predict_risk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--full-precision", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(4)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        lookup = {}
        for p in args.images.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if p.name in lookup:
                    raise ValueError(f"Duplicate image filename: {p.name}; supply a narrower image root")
                lookup[p.name] = p
        paths = [lookup[row["image"]] for row in manifest]
    else:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        parser.error("No images found")
    model = ImageRiskModel(args.models, args.configs, args.device, amp=not args.full_precision)
    features, detections = [], []
    for begin in range(0, len(paths), args.batch_size):
        images = []
        for path in paths[begin:begin+args.batch_size]:
            with Image.open(path) as image:
                images.append(to_tensor(image.convert("RGB")))
        concept, pred = model.extract(images)
        features.append(concept); detections.extend(pred)
        print(f"images={min(begin+args.batch_size,len(paths))}/{len(paths)}", flush=True)
    concept = torch.cat(features)
    risk = predict_risk(concept, args.models, model.config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = [str(path.relative_to(args.images)) for path in paths]
    torch.save({"images":names,"concept":concept,"risk":risk,"detections":detections}, args.output)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["image", "predicted_missed_objects"])
        for index in torch.argsort(risk,descending=True,stable=True).tolist():
            writer.writerow([names[index],float(risk[index])])


if __name__ == "__main__":
    main()
