# skyseg (U²-Net)

Automatic sky segmentation for darktable's one-click "ai sky" mask. One pass
over the whole frame gives a soft sky mask, with no click prompts. This is
the MIT-licensed sky model; [SkyAR](../mask-ai-sky-skyar/README.md) does the
same task faster, but only for non-commercial use.

## Source

- Repository: <https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing>
- Architecture paper: [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007) (Pattern Recognition 2020); the sky model has no paper of its own
- License: MIT (the U²-Net architecture is Apache-2.0)
- ONNX weights: [JianyuanWang/skyseg](https://huggingface.co/JianyuanWang/skyseg), a Hugging Face mirror of upstream's 167 MB export

## Architecture

U²-Net, about 44M parameters, trained by upstream for sky segmentation.
Upstream's ONNX keeps all seven of U²-Net's outputs: the fused map and the
six side maps it is built from, which only matter for training.
`convert.py` keeps the fused one, since darktable's AI mask runner takes
models with exactly one output, and renames the tensors. It checks that
the output it keeps is the 6-to-1 fusion convolution and that the cut graph
reproduces it exactly.

## ONNX Model

| Direction | Tensor       | Shape             | Type    |
| --------- | ------------ | ----------------- | ------- |
| in        | input_image  | 1 x 3 x 320 x 320 | float32 |
| out       | output_image | 1 x 1 x 320 x 320 | float32 |

Measured with `dtai validate`. FP32, 168 MiB.

### Preprocessing (client-side)

- resize source image to 320 × 320, stretching rather than cropping
- convert RGB to `[0, 1]` float
- normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- transpose to NCHW

### Postprocessing

- none for the values: the output is already a probability in `[0, 1]`,
  since U²-Net applies its sigmoid inside the graph. Applying another one
  would flatten the mask
- resize the mask back to the original image dimensions

## Quality

At 320 × 320 the mask is coarse: edges against foliage, petals and other
fine detail come out as soft blobs. Thin dark lines against the sky, such
as rigging, are partly caught. Upstream sharpens the mask with a separate
post-processing step; in darktable, the blend panel's feathering, guided by
the image, does a similar job. Upstream also notes that some building
detail is taken for sky, and that some textured clouds are missed.

Inference took about 1 s per image on CPU in `dtai demo` over the three
`samples/mask` images, against about 0.3 s for SkyAR.

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | MIT                                                                                                |
| OSAID v1.0               | No: the training data is not disclosed                                                             |
| MOF                      | Not classified: no paper or data card for the sky model                                            |
| Training data license    | Unknown                                                                                            |
| Training data provenance | Not disclosed by upstream                                                                          |
| Training code            | [U²-Net](https://github.com/xuebinqin/U-2-Net) (Apache-2.0), per upstream                          |
| Known limitations        | Coarse edges at 320 × 320; some building detail taken for sky; some textured clouds missed         |
| Published research       | Architecture only: [U²-Net](https://arxiv.org/abs/2005.09007) (Pattern Recognition 2020)           |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic sky segmentation; no clicks required                                                     |
| Reproducibility          | Partial: the ONNX is upstream's; `convert.py` only removes the side outputs                        |
