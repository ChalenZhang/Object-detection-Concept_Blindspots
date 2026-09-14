# Source Style Bank

The bank contains the photographic source view and twenty appearance transformations of each source scene. The source is [Cityscapes](https://www.cityscapes-dataset.com/), introduced in [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html).

| Group | Styles |
| --- | --- |
| Painting and fine art | Oil painting, sketching, line art, crayon drawing, Chinese painting |
| Animation and comics | Ghibli anime, American comics, chibi comics, painterly anime, pixel art |
| Film and games | 3D modeling, post-apocalyptic, science fiction, cyberpunk, AAA game scene |
| Materials and crafts | Paper cutting, stained glass, building blocks, collage, textile art |

The recorded bank was generated using commercial Seedream image-to-image models, spanning versions [3.0](https://arxiv.org/abs/2504.11346), [4.0](https://arxiv.org/abs/2509.20427), [4.5](https://seed.bytedance.com/en/seedream4_5), and [5.0 Lite](https://seed.bytedance.com/en/seedream5_0_lite). Generation starts from the source image for each style. Accepted views preserve the scene layout and the identity, count, position, and geometry of labeled objects, allowing the original labels to be reused locally. Views were compared with their source images and rejected when correspondence was visibly broken.

For local generation, the appearance-editing instruction can be expressed as:

```text
Render the supplied image in <STYLE>. Change the rendering medium, texture,
and palette while retaining the original scene, camera viewpoint, image
dimensions, and the number, positions, and shapes of all objects.
```

This describes the editing intent rather than a deterministic generation seed. Commercial model versions and stochastic outputs do not guarantee pixel-identical regeneration. Use the released model package for image inference and the statistics package for the recorded downstream experiments.

Obtain source images and labels directly under the [Cityscapes terms](https://www.cityscapes-dataset.com/license/). Generated views remain local. Store each view as `<style>/images/<scene>.<extension>`; retain the common scene identifier and only reuse labels after checking geometric correspondence. The image editor is not invoked during detection or risk inference.
