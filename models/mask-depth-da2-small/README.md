# Depth Anything V2 Small

Monocular depth estimation: one photograph in, a relative depth map out,
bright where the scene is near and dark where it is far, sharp along
object edges. It can drive depth-based masks, such as selecting the
foreground or the distance. The ONNX is the onnx-community export of the
official checkpoint, used as it is.

## Source

- Repository: <https://github.com/DepthAnything/Depth-Anything-V2>
- Paper: [Depth Anything V2](https://arxiv.org/abs/2406.09414) (NeurIPS 2024)
- License: Apache-2.0 (Small only; Base, Large and Giant are CC-BY-NC-4.0)
- ONNX weights: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)

## Architecture

A DINOv2-Small encoder with a DPT (Dense Prediction Transformer) decoder,
24.8M parameters. A large teacher model trained on synthetic data labels
62M real images, and this student is trained on those labels.

## ONNX Model

| Direction | Tensor          | Shape         | Type    |
| --------- | --------------- | ------------- | ------- |
| in        | pixel_values    | 1 x 3 x H x W | float32 |
| out       | predicted_depth | 1 x H x W     | float32 |

Measured with onnxruntime. FP32, 94 MiB, opset 14. H and W are dynamic:
the output is the input size rounded down to a multiple of 14, the ViT
patch size.

### Preprocessing (client-side)

As upstream's `preprocessor_config.json`:

- keep the aspect ratio, bring the short side to 518 and both sides to
  multiples of 14 (bicubic)
- convert RGB to `[0, 1]` float
- normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- transpose to NCHW

Stretching the image to a 518 square also runs, but with less detail:
thin structures such as overhead wires survive only at the kept aspect.

### Postprocessing

- the output is relative inverse depth (disparity), unbounded and with a
  scale and offset of its own per image: normalize it per image
- resize the map back to the original image dimensions

The values rank what is nearer; they are not distances in meters.

## Performance

About 0.2 s at 518 x 518 and 0.4 s at 784 x 518 (a 3:2 image at the
upstream size) with the CPU provider on an Apple silicon laptop.

## Selection Criteria

| Property                 | Value                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Model license            | Apache-2.0                                                                                                 |
| OSAID v1.0               | Open Weights                                                                                               |
| MOF                      | Class II (Open Tooling)                                                                                    |
| Training data license    | Mixed (SA-1B: Meta research-only; Places365: non-commercial; others: various open)                         |
| Training data provenance | Teacher: [Hypersim](https://github.com/apple/ml-hypersim), [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/), [TartanAir](https://theairlab.org/tartanair-dataset/), [BlendedMVS](https://github.com/YoYo000/BlendedMVS), IRS (595K). Student: [BDD100K](https://www.bdd100k.com/), [Google Landmarks](https://github.com/cvdfoundation/google-landmark), [ImageNet-21K](https://www.image-net.org/), [LSUN](https://www.yf.io/p/lsun), [Objects365](https://www.objects365.org/), [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), [Places365](http://places2.csail.mit.edu/), [SA-1B](https://ai.meta.com/datasets/segment-anything/) (62M). DINOv2-Small on LVD-142M (not public) |
| Training code            | [Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2)                                          |
| Known limitations        | LVD-142M: 142M web-crawled images, not released by Meta, not auditable. SA-1B: research-only (not OSI). Places365: non-commercial research terms. Relative depth only, not metric |
| Published research       | [Depth Anything V2](https://arxiv.org/abs/2406.09414) (NeurIPS 2024)                                      |
| Inference                | Local only, no cloud dependencies                                                                          |
| Scope                    | Monocular relative depth estimation                                                                        |
| Reproducibility          | Pre-converted ONNX of the official checkpoint; demo pipeline provided                                      |
