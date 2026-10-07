# SkyAR sky matting

Automatic sky segmentation for darktable's one-click "ai sky" mask, using
the sky matting network from SkyAR. One pass over the whole frame gives a
soft sky matte, with no click prompts.

> **Non-commercial use only.** SkyAR is licensed CC BY-NC-SA 4.0, and the
> images it was trained on are licensed for non-commercial research and
> education only. Do not use this model on work you are paid for or sell.
> [skyseg](../mask-ai-sky-skyseg/README.md) is the MIT-licensed model for
> the same task.

## Source

- Repository: <https://github.com/jiupinjia/SkyAR>
- Paper: [Castle in the Sky: Dynamic Sky Replacement and Harmonization in Videos](https://arxiv.org/abs/2010.11800) (IEEE TIP 2022)
- License: CC BY-NC-SA 4.0
- Weights: upstream's `checkpoints_G_coord_resnet50.zip` on Google Drive, exported here

## Architecture

`coord_resnet50`: a ResNet-50 encoder with an FPN-style decoder, where the
first convolution and every decoder convolution take an extra row-position
channel (CoordConv). About 50.5M parameters, ending in a sigmoid.

`convert.py` builds the network from `vendor/SkyAR/networks.py`, loads the
checkpoint's `model_G_state_dict` and exports at 384 × 384, the size SkyAR
was trained and run at. It stubs out two imports that `networks.py` makes
without using them for the network (SkyAR's `utils`, which needs a
scikit-image function since removed, and matplotlib), and skips the
ImageNet weight download the constructor asks for, since the checkpoint
overwrites those weights. The export is checked against PyTorch.

## ONNX Model

| Direction | Tensor       | Shape             | Type    |
| --------- | ------------ | ----------------- | ------- |
| in        | input_image  | 1 x 3 x 384 x 384 | float32 |
| out       | output_image | 1 x 1 x 384 x 384 | float32 |

Measured with `dtai validate`. FP32, 185 MiB; opset 17.

### Preprocessing (client-side)

- resize source image to 384 × 384, stretching rather than cropping
- convert RGB to `[0, 1]` float, with no mean or std normalization:
  SkyAR feeds the network plain `[0, 1]` pixels
- transpose to NCHW

### Postprocessing

- none for the values: the output is already a matte in `[0, 1]`, since
  the network ends in a sigmoid. Applying another one would flatten it
- resize the matte back to the original image dimensions

## Quality

The raw matte is softer than skyseg's and misses thin lines against the
sky, such as rigging. Upstream does not use it raw: it sharpens it with a
guided filter (radius 20, guided by the blue channel). In darktable, the
blend panel's feathering, guided by the image, does a similar job.

Inference took about 0.3 s per image on CPU in `dtai demo` over the three
`samples/mask` images, against about 1 s for skyseg.

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | CC BY-NC-SA 4.0 (non-commercial)                                                                   |
| OSAID v1.0               | No: the license restricts use                                                                      |
| MOF                      | Not classified: the license is not an open one                                                     |
| Training data license    | ADE20K images: non-commercial research and education only                                          |
| Training data provenance | Sky mattes by Liba et al. (CVPRW 2020) over ADE20K, subset ADE20K+DE+GF: 9187 training and 885 validation images |
| Training code            | [SkyAR](https://github.com/jiupinjia/SkyAR) (CC BY-NC-SA 4.0)                                      |
| Known limitations        | Non-commercial only; soft edges in the raw matte; thin lines against the sky missed                |
| Published research       | [Castle in the Sky](https://arxiv.org/abs/2010.11800) (IEEE TIP 2022)                              |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic sky segmentation; no clicks required                                                     |
| Reproducibility          | Full pipeline: upstream checkpoint and network code, exported by `convert.py`                      |
