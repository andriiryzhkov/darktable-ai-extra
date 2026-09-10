# BiRefNet Lite

Automatic subject / foreground segmentation. Produces a soft alpha mask of the
dominant subject in a single forward pass – no point prompts required,
unlike the click-based SAM 2.1 and SegNext models in this repo.

## Source

- Repository: <https://github.com/ZhengPeng7/BiRefNet>
- Paper: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407) (CAAI AIR 2024)
- License: MIT
- ONNX weights: [onnx-community/BiRefNet_lite-ONNX](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX)

## Architecture

BiRefNet "lite" – Swin Tiny (`swin_v1_t`) backbone with the BiRefNet bilateral
reference head. Trained primarily for dichotomous image segmentation (DIS) and
salient object detection.

## ONNX Model

| Direction | Tensor       | Shape               | Type    |
| --------- | ------------ | ------------------- | ------- |
| in        | input_image  | 1 x 3 x 1024 x 1024 | float32 |
| out       | output_image | 1 x 1 x 1024 x 1024 | float32 |

Measured with `dtai validate`. The FP16 package stores half-precision
weights but keeps float32 at the graph boundary and casts internally, so a
caller feeds and reads float32 whichever variant is shipped.

Default ship is FP16 (109 MiB). The FP32 variant (214 MiB) lives at the same
HF path under `onnx/model.onnx`; swap the URL in `model.yaml` to fetch it
instead. Tensor names and shapes are identical between the two.

### Preprocessing (client-side)

- resize source image to 1024 × 1024 (LANCZOS)
- convert RGB to `[0, 1]` float
- normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- transpose to NCHW and cast to the model's expected dtype

### Postprocessing

- apply sigmoid (the ONNX file outputs logits) to get a `[0, 1]` mask
- resize the mask back to original image dimensions

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | MIT                                                                                                |
| OSAID v1.0               | Open Source AI                                                                                     |
| MOF                      | Class I (Open Science)                                                                             |
| Training data license    | DIS5K: Apache 2.0 (dataset license)                                                                |
| Training data provenance | [DIS5K](https://github.com/xuebinqin/DIS) – 5470 high-resolution images collected from Flickr and manually annotated for dichotomous image segmentation |
| Training code            | [MIT](https://github.com/ZhengPeng7/BiRefNet)                                                      |
| Known limitations        | DIS5K source images come from Flickr; per-image creator consent is not documented. Curated and manually annotated rather than a LAION-style web scrape, but worth flagging under the strictest reading of the AI policy |
| Published research       | [BiRefNet](https://arxiv.org/abs/2401.03407) (CAAI AIR 2024)                                       |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic subject / foreground segmentation; no clicks required                                    |
| Reproducibility          | Full pipeline – ONNX is downloaded from the `onnx-community` mirror of the official BiRefNet_lite checkpoint |
