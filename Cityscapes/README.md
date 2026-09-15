# Cityscapes 20-Style Dataset

We thank the Cityscapes team and contributors for creating and maintaining this valuable resource for urban scene understanding. We respect their work and the conditions under which it is shared. Our style bank builds on [Cityscapes](https://www.cityscapes-dataset.com/), introduced in [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html).

## License and Availability

The official [Cityscapes Terms and Conditions](https://www.cityscapes-dataset.com/license/) include two relevant restrictions:

1. **License agreement, item 3:** "That you do not distribute this dataset or modified versions."
2. **Terms of Use, Section 4.2:** protected dataset contents must not be made accessible to third parties. This also covers modified or derived works from which the dataset can be reconstructed or derived.

Our style-transferred images are modified Cityscapes images, so we cannot directly distribute the 20-style image bank under these terms. Original images and inherited annotations must also be obtained through the official provider.

> **Dataset download: Coming soon, subject to authorization.** We are actively seeking permission from the relevant rights holders to release the style-transferred dataset. Download information will be added here once authorization is granted.

The [model and statistics packages](../README.md#packages) are available now.

## Dataset Composition

The bank contains **59,488 accepted style-transferred views of 2,975 Cityscapes training images**. Each source scene is assigned 20 target styles, organized into four groups. The photographic source view, `ORIGIN`, is kept separately and is not one of the 20 styles.

| Group | Styles |
| --- | --- |
| Painting and fine art | Oil painting, sketching, line art, crayon drawing, Chinese painting |
| Animation and comics | Ghibli anime, American comics, chibi comics, painterly anime, pixel art |
| Film and games | 3D modeling, post-apocalyptic, science fiction, cyberpunk, AAA game scene |
| Materials and crafts | Paper cutting, stained glass, building blocks, collage, textile art |

The [style-name list](style_names.txt) provides the corresponding directory names. See the [same-scene overview](../README.md#appearance-transformations) for examples.

## How We Generated the Bank

We generated the bank offline using commercial Seedream image-to-image models across versions [3.0](https://arxiv.org/abs/2504.11346), [4.0](https://arxiv.org/abs/2509.20427), [4.5](https://seed.bytedance.com/en/seedream4_5), and [5.0 Lite](https://seed.bytedance.com/en/seedream5_0_lite). Each generation takes the original photograph and a style instruction as input. Every style starts from the original, rather than from a previously stylized image.

The editing objective is to change appearance while retaining scene structure: rendering medium, surface texture, and palette may change, but object identities, counts, shapes, locations, and camera framing must remain aligned. Each generated image was manually compared with its original. Outputs with altered objects or inconsistent annotation alignment were regenerated and checked again. Accepted views inherit the original bounding boxes and classes, with the source scene identifier retained for paired analysis. No image generator is used during detection or risk inference.

## Generate a Similar Style Bank

The same original-image-plus-style-prompt workflow can be used with capable image editors such as:

| Model | Editing route |
| --- | --- |
| [Seedream 5.0 Lite](https://seed.bytedance.com/en/seedream5_0_lite) | Image-conditioned editing with the original scene and a style prompt |
| [FLUX.2](https://docs.bfl.ai/flux_2/flux2_image_editing) | Single-reference image editing with explicit scene-preservation instructions |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Locally deployable image editing with publicly available model weights |

Use the prompt below, tune the editing strength on a small pilot set, and manually review the outputs to construct a similar multi-style bank. Obtain source images through the [Cityscapes download portal](https://www.cityscapes-dataset.com/downloads/). Use local editing for restricted images; hosted editing requires permission covering disclosure to the provider.

### Prompt Template

Replace `<STYLE>` with a style from the table, using spaces rather than directory-name hyphens:

```text
Restyle this input image as <STYLE>. Apply the style through colors,
surface textures, and rendering technique only. Retain the exact scene
composition and camera view. Keep every object in its original location,
with its original size, outline, and identity. Do not introduce, remove,
duplicate, or rearrange objects. Preserve the road layout, occlusions,
image boundaries, and output dimensions. Do not crop or add padding.
```

For example, set `<STYLE>` to `Oil painting` and add `Use visible brushwork and painted surfaces while keeping object boundaries intact.` For `Cyberpunk`, add `Express the style through the existing surfaces and color palette; do not add signs, buildings, vehicles, or people.`

### Generation and Quality Checks

1. **Prepare the source.** Keep the Cityscapes training scenes separate from evaluation scenes, and retain each image's original filename and annotations.
2. **Tune on a small pilot.** Start with the editor's default image-editing settings and the original aspect ratio. Where available, increase reference adherence or reduce editing strength if geometry drifts; strengthen the texture and palette instructions if the style is too weak.
3. **Generate each style independently.** Supply the original image for every entry in `style_names.txt`. Preserve the original output dimensions.
4. **Check every output.** Compare it with the original and overlay the source boxes. Inspect small or partly occluded objects, object counts and outlines, box alignment, road geometry, framing, and image readability. Reject and regenerate outputs with missing, added, duplicated, displaced, or deformed objects, incorrect framing, or corrupted files.
5. **Keep accepted pairs.** Reuse source labels only after the alignment check. Store the accepted view with the same scene identifier in the corresponding style directory.

Suggested local layout:

```text
Cityscapes-Multi-Style/
  ORIGIN/
    images/<scene>.<ext>
    labels/<scene>.txt
  Oil-Painting/
    images/<scene>.<ext>
    labels/<scene>.txt
  <other-style>/
    images/<scene>.<ext>
    labels/<scene>.txt
```

Here, `labels/` contains locally prepared detection annotations. Generation and manual curation are completed before concept analysis and risk-model training.
