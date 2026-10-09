# IS-Net (general use)

Fast automatic subject / foreground segmentation for darktable's one-click
"ai subject" mask. One pass over the whole frame gives a soft mask of the
dominant subject, with no click prompts, like
[BiRefNet](../mask-ai-subject-birefnet/README.md) and
[BiRefNet Lite](../mask-ai-subject-birefnet-lite/README.md), but about
twenty times faster than BiRefNet on CPU.

Pick it for speed on simple subjects. Its masks are less solid than
BiRefNet's: it finds the right subject, but tends to leave grey or
see-through patches inside it. Pick BiRefNet when the mask must be solid.

## Source

- Repository: <https://github.com/xuebinqin/DIS>
- Paper: [Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2203.03041) (ECCV 2022)
- License: Apache-2.0
- Weights: `isnet-general-use.pth`, released by upstream on 2022-08-17 as
  "the optimized model for general use of our IS-Net"
- ONNX: the export published with [rembg](https://github.com/danielgatis/rembg) (MIT)

## Architecture

IS-Net, about 44M parameters: the dichotomous image segmentation network
from the authors of the DIS5K dataset, which BiRefNet was later built to
improve on. rembg's ONNX keeps all twelve of IS-Net's outputs: six sigmoid
side maps, finest first, and the six feature maps they are computed from,
which only matter for training. `convert.py` keeps the first side map, the
one upstream's `Inference.py` uses as the result, since darktable's AI mask
runner takes models with exactly one output. It checks that the output it
keeps is IS-Net's `side1` head and that the cut graph reproduces it
exactly.

## ONNX Model

| Direction | Tensor       | Shape               | Type    |
| --------- | ------------ | ------------------- | ------- |
| in        | input_image  | 1 x 3 x 1024 x 1024 | float32 |
| out       | output_image | 1 x 1 x 1024 x 1024 | float32 |

Measured with `dtai validate`. FP32, 170 MiB.

### Preprocessing (client-side)

- resize source image to 1024 × 1024, stretching rather than cropping
- convert RGB to `[0, 1]` float
- subtract 0.5 with no scaling: mean `[0.5, 0.5, 0.5]`, std `[1.0, 1.0, 1.0]`,
  as upstream's `Inference.py` does
- transpose to NCHW

### Postprocessing

- none for the values: the output is already a probability in `[0, 1]`,
  since IS-Net applies its sigmoid inside the graph. Applying another one
  would flatten the mask. Upstream also stretches the result to its
  minimum and maximum; on the sample images the output already spans
  0 to 1, so nothing is lost without it
- resize the mask back to the original image dimensions

## Quality

On the DIS5K test sets, IS-Net's error is about twice BiRefNet's: the
BiRefNet paper (Table 4) gives a mean absolute error of .070 against .035
and a weighted F-measure of .726 against .858 over the 2,000 test images.
Those figures are for upstream's academic weights, trained on DIS5K V1.0,
not for these general-use ones.

On photos the difference shows inside the subject more than at its edge.
On a cat or a stuffed toy against a plain background, the mask is close to
BiRefNet's. On two trams in a street it finds both, but leaves their
windows grey; on a ship's bow with rigging it traces the ladder and the
lines but leaves the hull mostly out. darktable refines the mask's edges
from the image, which cannot fill such patches.

Inference took about 0.7 s per image on CPU in `dtai demo` over the three
`samples/mask` images. BiRefNet took about 10 s per run on the same CPU,
measured separately.

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | Apache-2.0                                                                                         |
| OSAID v1.0               | No: the training data of the general-use weights is not disclosed                                  |
| MOF                      | Not classified: no data card for the general-use weights                                           |
| Training data license    | Unknown for these weights; DIS5K's terms are in upstream's `DIS5K-Dataset-Terms-of-Use.pdf`        |
| Training data provenance | Not stated for the general-use weights, which upstream calls "not DIS V2.0"; the academic IS-Net was trained on DIS5K V1.0 |
| Training code            | [DIS](https://github.com/xuebinqin/DIS) (Apache-2.0)                                               |
| Known limitations        | Grey or see-through patches inside the subject; weaker than BiRefNet on busy scenes                |
| Published research       | [IS-Net](https://arxiv.org/abs/2203.03041) (ECCV 2022)                                             |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic subject / foreground segmentation; no clicks required                                    |
| Reproducibility          | Partial: the ONNX is rembg's export of upstream's checkpoint; `convert.py` only removes the training outputs |
