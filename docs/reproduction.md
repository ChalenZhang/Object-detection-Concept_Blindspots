# Reproduction

## Model Package

`detector_r101.pt` contains the frozen Faster R-CNN ResNet-101-FPN detector used for image feature extraction. `detsae.pt` contains the class-specific sparse autoencoders, normalization tensors, and learned decoder directions. `risk_seed_2027.pt`, `risk_seed_2028.pt`, and `risk_seed_2029.pt` contain the selected risk predictors. All checkpoints load with `torch.load(..., weights_only=True)`.

Each of 101 clusters contributes four image summaries: maximum activation, mean of the three largest activations, and the corresponding confidence-weighted values. The 17 blindspots retain their individual identities through an additional copy of their 68 channels. The MLP maps the resulting 472 inputs through layers of width 256 and 128 with GELU and dropout 0.1. Its three heads learn missed-object occurrence, log missed-object count, and log detection-quality deficit for five class groups. Training combines binary cross-entropy, smooth-L1 count loss, quality loss weighted by 0.5, and a pairwise ranking loss weighted by 0.2.

Inference averages the predicted Car+Person missed-object counts across three risk predictors. The score is a ranking signal, not a calibrated probability. The same score is used on all domains, including the Car-only SIM10K endpoint.

## Statistics Package

| File | Numerical content | Reproduction |
| --- | --- | --- |
| `source_train.pt` | 62,463 image-level, 404-channel pooled summaries, five-group training targets, scene indices | Risk-MLP training |
| `source_calibration.pt` | 500 source calibration summaries and targets | Checkpoint selection |
| `discovery.pt` | All cluster memberships in the regression design, standardized nuisance covariates, paired quality values, valid-view masks, scene indices, and fixed CV folds | Joint blindspot discovery |
| `targets/*.pt` | Per-image pooled summaries, missed counts, and frozen comparison scores | Six-domain risk ranking |
| `interventions/*.csv.gz` | Before/after classification outputs, matched control identifiers, doses, and scene indices | Intervention summaries |

These statistics reproduce the downstream numerical analysis, not a pixel-identical regeneration of the commercially produced style images. They contain no spatial feature maps or per-object RoI feature bank. Inference on locally obtained images uses the complete model package.

## Intervention Records

All 17 frozen R101 blindspots and all 16 frozen R50 blindspots are retained. For an eligible paired object, the intervention moves one cluster's contribution toward its ORIGIN value while preserving the remaining feature residual and all other activations. Excess events have a styled-to-source contribution projection ratio at least 1.5; missing-activation events have a ratio at most 0.5. The four doses are 0.25, 0.5, 0.75, and 1.0. Opposite, shuffled-source, normal-cluster, and random directions are matched in feature-change norm.

`summarize_interventions.py` reports every cluster, mechanism, dose, and control, including negative effects. Margin gain is the change in the correct-class logit minus its strongest competitor. Top-1 net recovery is the fraction changing from wrong to correct minus the fraction changing from correct to wrong. The full-dose paired restoration includes 5,000 scene-bootstrap confidence intervals. These are frozen-head diagnostic results, not detector mAP improvements or an automatic image-level repair method.

## Blindspot Test

The object-level quality is the score times IoU of a correct-class, one-to-one IoU50 match, or zero if unmatched. For each style, subtract the class-wide mean ORIGIN-minus-style loss. Jointly regress each object's average centered loss on its fixed concept identities, controlling for source geometry and source quality. Select ridge strength using the supplied five scene folds. A blindspot must have an effect and scene-bootstrap 95% lower bound above 0.05, pass joint BH-FDR 0.05, and remain above 0.05 in every leave-one-style-out fit. The default run uses 2,000 scene-bootstrap replicates.

## Risk Metrics

An image is positive if it contains at least one missed evaluation object. A correct match requires the correct class, confidence at least 0.05, and IoU at least 0.5. Matching is one-to-one, with at most 100 predictions per class.

- **Capture@5%:** missed objects in the highest-risk `ceil(0.05 * N)` images divided by missed objects in all `N` images.
- **AUROC:** probability that a randomly chosen positive image ranks above a negative image, with half credit for tied scores.
- **AUPRC:** average precision of the image-level failure ranking. It is not detection mAP.
- **NAURC:** normalized area under the risk-coverage curve; lower is better. Accept images in increasing risk order. With failure prevalence `p`, `AURC` is the mean cumulative accepted failure rate, and `NAURC = (AURC - A*) / (p - A*)`, where `A* = p + (1-p) log(1-p)` is the continuous oracle reference.

The saved benchmark scores use deterministic empirical rank percentiles. Sorting is stable. Average precision is computed as precision at each positive rank, weighted uniformly over positive images. Metrics with no applicable positive/negative class are reported as undefined.

## Numerical Settings

The recorded environment uses Python 3.12 and the package versions in `requirements.txt`. Source training summaries were extracted in full precision; target evaluation uses mixed-precision detector inference and stores pooled summaries in float16. Use image batch size 2 for the supplied target manifests. Different padding batches, accelerator kernels, or precision settings can slightly change proposal selection and rankings.

Risk training samples 64 distinct source scenes per update and one available view per scene. The three learning-rate candidates are 0.000125, 0.00025, and 0.0005, with cosine decay and weight decay 0.0001. `configs/training.json` records the validation schedule and the selected checkpoints. A short execution check can be run with `python scripts/train_risk.py --steps 2 --seeds 2027 --device cpu`; this does not reproduce the full training result.
