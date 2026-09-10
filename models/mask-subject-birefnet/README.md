# BiRefNet (Swin Large)

Automatic subject / foreground segmentation. Same single-pass workflow as
[BiRefNet Lite](../mask-subject-birefnet-lite/README.md) – produces a soft alpha
mask of the dominant subject without point prompts – but uses the full
Swin Large backbone (~4× the parameters) for finer boundary detail on hair,
fur and foliage.

Pick BiRefNet over Lite when edge precision matters more than runtime; pick
Lite when you want the fast interactive default.

## Source

- Repository: <https://github.com/ZhengPeng7/BiRefNet>
- Paper: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407) (CAAI AIR 2024)
- License: MIT
- ONNX weights: [onnx-community/BiRefNet-ONNX](https://huggingface.co/onnx-community/BiRefNet-ONNX)

## Architecture

BiRefNet with `swin_v1_l` (Swin Large) backbone and the BiRefNet bilateral
reference head. ~200M parameters total. Trained on DIS5K for dichotomous
image segmentation; works directly as a salient object detector / single-
subject foreground extractor.

## ONNX Model

| Direction | Tensor       | Shape               | Type    |
| --------- | ------------ | ------------------- | ------- |
| in        | input_image  | 1 x 3 x 1024 x 1024 | float32 |
| out       | output_image | 1 x 1 x 1024 x 1024 | float32 |

Measured with `dtai validate`. The FP16 package stores half-precision
weights but keeps float32 at the graph boundary and casts internally, so a
caller feeds and reads float32 whichever variant is shipped.

Default ship is FP16 (467 MiB). The FP32 variant (928 MiB) lives at the same
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

## Trade-off vs. Lite

| Metric                          | Lite (swin_v1_t)    | BiRefNet (swin_v1_l) |
| ------------------------------- | ------------------- | -------------------- |
| Parameters                      | ~44M                | ~200M                |
| FP16 ONNX size                  | 109 MiB             | 467 MiB              |
| CPU inference at 1024²         | ~6.4 s              | ~10.3 s              |
| Edge detail on hair / fur       | adequate            | noticeably better    |

Inference measured with `dtai demo` over the three `samples/mask` images on
one desktop CPU, so treat it as a ratio rather than an absolute – the gap is
closer to 1.6× than the 3–4× the parameter count suggests, because both are
dominated by the fixed 1024² input.

Reach for this variant when Lite is falling short on a specific class of
images. Note that darktable's own mask post-processing (joint-bilateral
upsampling, `common/ai/segmentation.c:195`) is not in play here: that path
only handles the sam, sam2 and segnext arches, so a Lua caller applies its
own refinement or none.

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
| Reproducibility          | Full pipeline – ONNX is downloaded from the `onnx-community` mirror of the official BiRefNet checkpoint |
