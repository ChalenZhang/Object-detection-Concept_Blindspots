# RealDriveSim 6,000-Image Subset

This release contains the complete extracted pool of original RealDriveSim images used by this study, not the full RealDriveSim dataset and not its style-transferred variants.

| Component | Images |
| --- | ---: |
| Day | 2,520 |
| Adverse-A | 2,580 |
| Adverse-B | 900 |
| Total | 6,000 |

The images are stored as 2048 x 1024 PNG files. [subset.csv](subset.csv) lists all image-label pairs, their component, sequence, and frame. [files.json](files.json) lists the downloadable ZIP parts and their contents. The Day / Adverse-A / Adverse-B names identify our extracted components; the official download page groups the source data into normal weather and two adverse-weather batches.

## Download and Use

Download every `realdrivesim-6000-part*.zip` file from the repository's **RealDriveSim 6,000 Images** release. Each is an ordinary ZIP archive. Extract all parts to the same destination; they populate one `realdrivesim-6000/` directory. For example, from the cloned repository with GitHub CLI installed:

```bash
gh release download realdrivesim-6000-v1.0 \
  --pattern 'realdrivesim-6000-part*.zip' --dir downloads
for archive in downloads/realdrivesim-6000-part*.zip; do
  unzip -n "$archive" -d datasets
done
```

The combined directory contains:

```text
realdrivesim-6000/
  images/                 6,000 original-view PNG files
  labels/                 6,000 corresponding YOLO label files
  annotations.json        COCO-style detection annotations
  classes.txt             YOLO class IDs and normalized names
  subset.csv              Complete 6,000-image list
  evaluation.json         Image order for the released evaluation
  files.json              ZIP part inventory
  README.md
```

YOLO rows are `class_id center_x center_y width height`, with coordinates normalized by image dimensions. Class IDs are 0 bicycle, 1 bus, 2 car, 3 motorcycle, 4 person, 5 rider, and 6 truck. The original label abbreviations `Motor` and `Psn.` are written as `motorcycle` and `person` in the category metadata. In `annotations.json`, category IDs are the YOLO IDs plus one; boxes use absolute `[left, top, width, height]` coordinates. Conversion follows the existing evaluation preprocessing, including removal of boxes with width or height at most one pixel.

To reproduce the released evaluation, use its existing image manifest rather than substituting the complete pool:

```bash
python code/scripts/predict.py --images datasets/realdrivesim-6000/images \
  --manifest data/splits/realdrivesim.json --output outputs/realdrivesim.pt
python code/scripts/evaluate_predictions.py --predictions outputs/realdrivesim.pt \
  --annotations datasets/realdrivesim-6000/annotations.json
```

To obtain predictions for all 6,000 images, omit `--manifest`.

## Source and License

**RealDriveSim: A Realistic Multi-Modal Multi-Task Synthetic Dataset for Autonomous Driving**, Arpit Jadon, Haoran Wang, Phillip Thomas, Michael Stanley, S. Nathaniel Cibik, Rachel Laurat, Omar Maher, Lukas Hoyer, Ozan Unal, and Dengxin Dai. IEEE Intelligent Vehicles Symposium, 2025. [Paper](https://arxiv.org/abs/2506.16319) | [Original dataset](https://realdrivesim.github.io/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Credit for the source images and annotations belongs to the RealDriveSim authors. This release selects and repackages an extracted image pool, with detection labels converted to a seven-class YOLO representation and an additional COCO-style JSON. The packaged PNG images and YOLO label files are copied without modification from the original-view pool used in our experiments; no resizing, recompression, or style transfer is applied during packaging. Category names are standardized in the accompanying metadata. This redistribution does not imply endorsement by the dataset authors. The images and derived detection annotations remain under CC BY 4.0, without additional use restrictions.
