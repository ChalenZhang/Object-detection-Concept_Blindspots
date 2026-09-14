# Data Preparation and Evaluation

Dataset providers and original papers are linked in the [dataset table](../README.md#datasets). Download images and annotations from the providers, not from this repository.

## Evaluation Subsets

| Manifest | Selection | Images |
| --- | --- | ---: |
| `splits/bdd100k.json` | Validation images with `timeofday=daytime` and `weather=clear` | 1,764 |
| `splits/kitti.json` | Labeled object-detection training split, used only for evaluation | 7,481 |
| `splits/realdrivesim.json` | Retained images from the extracted Day / Adverse-A / Adverse-B pool | 5,493 |
| `splits/sim10k.json` | All labeled images | 10,000 |
| `splits/foggy.json` | Validation scenes, beta 0.02 | 500 |
| `splits/rainy.json` | Alpha 0.02; 12 rain textures for each of 33 scenes | 396 |

The manifests preserve the exact row order used for evaluation. RealDriveSim was drawn from a 6,000-image pool containing 2,520 Day, 2,580 Adverse-A, and 900 Adverse-B images; the manifest specifies the images retained by the recorded evaluation. Rainy and Foggy are appearance-shift tests on source scenes, not independent-scene benchmarks. Rainy views of one scene share a scene index.

## Local Image Inference

Place the chosen dataset images under a directory. The prediction script searches recursively and resolves filenames against the selected manifest:

```bash
python scripts/predict.py --images /path/to/kitti/image_2 \
  --manifest data/splits/kitti.json --output outputs/kitti.pt
```

## Detection Annotations

The evaluation script accepts a local detection JSON with the following standard fields:

```json
{
  "images": [{"id": 1, "file_name": "example.png"}],
  "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [100, 80, 60, 40], "iscrowd": 0}]
}
```

Boxes use absolute pixel coordinates `[left, top, width, height]`. Image filenames must be unique within the selected manifest. Category IDs are mapped by names, rather than assumed to match detector IDs. `Pedestrian` and `Person_sitting` map to `person`; `Cyclist` maps to `rider`. Other aliases are specified in the evaluator. Crowd annotations are excluded.

```bash
python scripts/evaluate_predictions.py --predictions outputs/kitti.pt \
  --annotations /path/to/kitti_annotations.json
```

Evaluation uses Car and Person, except SIM10K, whose distributed detection annotations cover Car. Select `--groups car` for that endpoint. The risk predictor itself remains unchanged.

The detector's internal IDs are background 0, bicycle 1, bus 2, car 3, motorcycle 4, person 5, rider 6, and truck 7. Use the original label semantics when converting annotations: KITTI pedestrians must not be silently dropped, and unannotated SIM10K people must not be counted as negative examples.
