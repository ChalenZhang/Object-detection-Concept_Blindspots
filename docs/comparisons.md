# Comparison Methods

The statistics package contains detection-risk scores measured with a shared frozen detector and evaluation cohorts. Our method uses original and generated source views; external methods use original source images. Gradient-trained heads use 20,000 updates, batch size 64, and source-side checkpoint selection.

| Score ID | Method | Original work |
| --- | --- | --- |
| `sf` | SF | [Per-Frame mAP Prediction for Continuous Performance Monitoring of Object Detection During Deployment](https://openaccess.thecvf.com/content/WACV2021W/AVV/html/Rahman_Per-Frame_mAP_Prediction_for_Continuous_Performance_Monitoring_of_Object_Detection_WACVW_2021_paper.html) |
| `lfr`, `lf_ash_p75` | LFR, LF-ASH-P75 | [Run-time Introspection of 2D Object Detection in Automated Driving Systems Using Learning Representations](https://doi.org/10.1109/TIV.2024.3385531) |
| `lfa` | LFA | [LFA: Layer Feature Attention for Run-Time Introspection of 2D Object Detectors in Automated Driving](https://arxiv.org/abs/2606.00372) |
| `saod_global` | SAOD | [Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration](https://openaccess.thecvf.com/content/CVPR2023/html/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.html) |
| `vilu_det` | ViLU | [ViLU: Learning Vision-Language Uncertainties for Failure Prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.html) |
| `bta_det` | BTA | [Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html) |
| `kgfp_det` | KGFP | [Knowledge-Guided Failure Prediction: Detecting When Object Detectors Miss Safety-Critical Objects](https://openaccess.thecvf.com/content/CVPR2026W/SAIAD/html/Zimmermann_Knowledge-Guided_Failure_Prediction_Detecting_When_Object_Detectors_Miss_Safety-Critical_Objects_CVPRW_2026_paper.html) |
| `gram` | GRAM | [Detecting Out-of-Distribution Examples with Gram Matrices](https://proceedings.mlr.press/v119/sastry20a.html) |
| `knn5` | KNN-5 | [Out-of-Distribution Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d.html) |
| `vim_residual_pca100` | Residual | [ViM: Out-of-Distribution With Virtual-Logit Matching](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html) |

LFR and LF-ASH-P75 are variants of one work. SAOD uses its image-level uncertainty score; Residual uses ViM's PCA residual component. ViLU and BTA use detection-based failure-prediction inputs. The statistics package provides comparison scores; the model package provides our inference chain.
