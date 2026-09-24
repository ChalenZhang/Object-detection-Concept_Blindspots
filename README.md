# Concept Blindspots in Object Detection

DetSAE extracts task-relevant concepts from a frozen object detector. Paired appearance changes reveal concept blindspots. Separate false-negative (FN) and false-positive (FP) predictors use these blindspots to rank images by detection-error risk.

## Repository

```text
code/
  concept_blindspots/   Model and analysis modules
  scripts/             Training, inference, and evaluation
  configs/             Detector, concepts, and training settings
  tests/               Unit tests
  requirements.txt     Dependencies
data/
  cityscapes/          Style-generation guide and overview image
  realdrivesim/        Subset guide and file inventory
  splits/              Evaluation image lists
  README.md            Data preparation and label formats
models/                Frozen detector, DetSAE, and risk MLPs
statistics/            Source features and evaluation records
```

## Cityscapes 20-Style Dataset

**Coming soon, subject to authorization.** The [Cityscapes license](https://www.cityscapes-dataset.com/license/) restricts redistribution of original and modified images. We are actively seeking permission to release our 20-style bank. The [Cityscapes guide](data/cityscapes/README.md) provides the specific terms, style composition, generation prompts, and quality checks.

## Models and Statistics

Models and statistics are included as files in the repository and are downloaded with `git clone`. No separate package download or extraction is required.

| Directory | Contents |
| --- | --- |
| `models/` | Frozen R101 detector, DetSAE, and three risk MLPs per error type |
| `statistics/` | Source features and FN/FP labels, concept tests, intervention records, and six-benchmark evaluation records |

The detector weights in `models/detector_r101/` are loaded together automatically. `risk_seed_*.pt` contains the FN predictors and `risk_fp_seed_*.pt` the FP predictors. Statistics include `source_train.pt`, `source_calibration.pt`, `discovery.pt`, `targets/*.pt`, and `interventions/*.csv.gz`.

## Setup and Inference

Use Python 3.12. Run the following commands from the repository root:

```bash
python -m pip install -r code/requirements.txt
python code/scripts/predict.py --task fn --images /path/to/images \
  --output outputs/fn.pt --device cuda
python code/scripts/predict.py --task fp --images /path/to/images \
  --output outputs/fp.pt --device cuda
```

Each command saves detections, pooled concepts, and risk scores to the chosen `.pt` file, with an adjacent CSV ranking images by predicted error count. To use a dataset's evaluation image list:

```bash
python code/scripts/predict.py --task fp --images /path/to/images \
  --manifest data/splits/bdd100k.json --output outputs/bdd100k_fp.pt
python code/scripts/evaluate_predictions.py --predictions outputs/bdd100k_fp.pt \
  --annotations /path/to/annotations.json --output outputs/bdd100k_fp_metrics.json
```

See [data preparation](data/README.md) for annotation formats and dataset selections.

## Source Analysis and Evaluation

```bash
OPENBLAS_NUM_THREADS=4 python code/scripts/discover_blindspots.py
python code/scripts/train_risk.py --task fn --device cuda
python code/scripts/train_risk.py --task fp --device cuda
python code/scripts/summarize_interventions.py
python code/scripts/evaluate_statistics.py --task both --models models --device cuda
```

Training saves selected models to `outputs/retrained_models/`. The last command evaluates saved and recomputed FN/FP scores on each benchmark, writing metrics as fractions to `outputs/risk_metrics.csv`. Use `--models outputs/retrained_models` to evaluate retrained models, or omit `--models` to evaluate saved scores only.

### Model and Training

The frozen Faster R-CNN R101-FPN detector supplies object features to DetSAE. Each of 101 concept clusters contributes four pooled image channels: maximum, top-three mean, and their confidence-weighted counterparts. The 17 blindspots contribute 68 additional identity-preserving channels. Separate 472-input MLPs with hidden widths 256 and 128, GELU, and dropout 0.1 predict FN or FP occurrence and log error count for five category groups. The FN predictor also learns log quality deficit. Training combines weighted binary cross-entropy, Smooth-L1 count loss, and pairwise ranking loss with weight 0.2; the additional FN quality loss has weight 0.5.

Risk training uses 20,000 AdamW updates per candidate, batch size 64 with distinct source scenes, and one available view per scene from original Cityscapes and its 20 styled variants. Source features remain fixed. Learning rates are 0.000125, 0.00025, and 0.0005 with cosine decay and weight decay 0.0001. Task-specific Capture@5% on 500 original source calibration images selects checkpoints, with AUROC and earlier updates breaking ties. [FN settings](code/configs/training.json) and [FP settings](code/configs/training_fp.json) specify seeds, loss weights, and selected checkpoints. Inference averages the three models' predicted Car+Person error counts. Target feature extraction uses mixed precision and image batch size 2.

### Blindspot and Intervention Analysis

The discovery script fits all concept-cluster effects jointly, accounting for source quality, geometry, and shared class-style difficulty. A blindspot has an effect and 95% lower confidence bound above 0.05, passes BH-FDR 0.05, and remains above 0.05 after omitting each style. It uses 2,000 scene-bootstrap resamples.

Intervention records cover all 17 R101 and 16 R50 blindspots. Each intervention moves one cluster toward its paired source activation while keeping other activations and the residual fixed. The summary reports signed classification-margin changes and net Top-1 recovery for each cluster, dose, and control at the frozen head.

### Metrics

A false negative is a ground-truth object without a correct-class detection at confidence >= 0.05 and IoU >= 0.5. A false positive is an unmatched detection at confidence >= 0.5, after descending-confidence, one-to-one same-class matching at IoU >= 0.5, retaining at most 100 detections per class. KITTI excludes unmatched predictions assigned to `Van`, `Person_sitting`, or `DontCare` using the [annotation instructions](data/README.md#detection-annotations).

- **Capture@5%:** errors of the selected type in the highest-risk `ceil(0.05 * N)` images divided by all errors of that type.
- **AUROC:** probability that an image containing the selected error type ranks above an image without it; tied scores receive half credit.
- **AUPRC:** average precision of the image-level failure ranking.
- **NAURC:** `(AURC - A*) / (p - A*)`, where `p` is image failure prevalence, `A* = p + (1-p) log(1-p)`, and AURC averages cumulative failure rates as images are accepted from lowest to highest risk. Lower is better.

## Appearance Transformations

![One source scene and its twenty style-transferred views](data/cityscapes/styles-overview.jpg?raw=1)

One [Cityscapes](https://www.cityscapes-dataset.com/) scene and its 20 style-transferred views. Source imagery belongs to the Cityscapes rights holders; the transformations were produced for this study. See the [generation guide](data/cityscapes/README.md).

Please report copyright, privacy, or other rights concerns through the repository's Issues page. Affected material will be promptly hidden or removed during review.

## Datasets

| Dataset | Original publication | Data |
| --- | --- | --- |
| Cityscapes | [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html) | [Website](https://www.cityscapes-dataset.com/), [20-style guide](data/cityscapes/README.md) |
| BDD100K | [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html) | [Website](http://bdd-data.berkeley.edu/), [Download guide](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst) |
| KITTI | [Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | [Website](https://www.cvlibs.net/datasets/kitti/) |
| RealDriveSim | [RealDriveSim: A Realistic Multi-Modal Multi-Task Synthetic Dataset for Autonomous Driving](https://arxiv.org/abs/2506.16319) | [Website](https://realdrivesim.github.io/), [subset guide](data/realdrivesim/README.md) |
| SIM10K | [Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?](https://arxiv.org/abs/1610.01983) | [Original dataset](https://doi.org/10.7302/e1f1-3d97) |
| Foggy Cityscapes | [Semantic Foggy Scene Understanding with Synthetic Data](https://arxiv.org/abs/1708.07819) | [Website](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) |
| Rainy Cityscapes | [Depth-Attentional Features for Single-Image Rain Removal](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.html) | [Website](https://www.cityscapes-dataset.com/downloads/) |

The [RealDriveSim subset guide](data/realdrivesim/README.md) describes the **6,000 original-view images: 2,520 Day, 2,580 Adverse-A, and 900 Adverse-B**, their labels, and evaluation usage. Images and labels are available from this repository's **RealDriveSim 6,000 Images** GitHub Release.

## Terms

Code is provided under the [MIT license](LICENSE). Checkpoints and statistics are for non-commercial research, subject to the original dataset terms. The [Cityscapes terms](https://www.cityscapes-dataset.com/license/) restrict image redistribution; see the [Cityscapes guide](data/cityscapes/README.md#license-and-availability). The overview illustration retains its source image rights and is outside the software license.

The RealDriveSim subset and its detection annotations are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), independently of the checkpoint and statistics terms. See the [attribution and processing notice](data/realdrivesim/README.md#source-and-license).
