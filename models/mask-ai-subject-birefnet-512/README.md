# BiRefNet 512 (Swin Large)

Faster automatic subject / foreground segmentation for darktable's
one-click "ai subject" mask. The same Swin Large network as
[BiRefNet](../mask-ai-subject-birefnet/README.md), trained by upstream at
512 × 512 instead of 1024 × 1024: a quarter of the pixels makes it about
four times faster.

Pick it for speed on scenes with one clear subject. Around the subject it
is the cleanest of the subject models here: it leaves the least mask on
what lies next to it. In busy scenes it can miss large parts of the
subject, where BiRefNet and [BiRefNet Lite](../mask-ai-subject-birefnet-lite/README.md)
do not.

## Source

- Repository: <https://github.com/ZhengPeng7/BiRefNet>
- Weights: [ZhengPeng7/BiRefNet_512x512](https://huggingface.co/ZhengPeng7/BiRefNet_512x512)
- Paper: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407) (CAAI AIR 2024)
- License: MIT
- ONNX weights: [onnx-community/BiRefNet_512x512-ONNX](https://huggingface.co/onnx-community/BiRefNet_512x512-ONNX)

## Architecture

BiRefNet with `swin_v1_l` (Swin Large) backbone and the BiRefNet bilateral
reference head, about 232M parameters, as in the full model. Upstream
trained it at 512 × 512 "for faster and more accurate lower resolution
inference". On DIS-VD at 512 × 512 upstream reports a maxF of .879 and an
MAE of .040, against .834 and .050 for the general BiRefNet run at the
same size; the BiRefNet paper gives .891 and .038 for the general model at
1024 × 1024.

## ONNX Model

| Direction | Tensor       | Shape             | Type    |
| --------- | ------------ | ----------------- | ------- |
| in        | input_image  | 1 x 3 x 512 x 512 | float32 |
| out       | output_image | 1 x 1 x 512 x 512 | float32 |

Measured with `dtai validate`. The FP16 package stores half-precision
weights but keeps float32 at the graph boundary and casts internally, so a
caller feeds and reads float32 whichever variant is shipped.

Default ship is FP16 (452 MiB). The FP32 variant (897 MiB) lives at the same
HF path under `onnx/model.onnx`; swap the URL in `model.yaml` to fetch it
instead. Tensor names and shapes are identical between the two. Their masks
agree to within 1/255 on all but 0.15% of the pixels of the sample images.
On CPU the FP32 variant ran about 20% faster, since the FP16 one pays for
its casts there; on a GPU FP16 is usually the faster.

### Preprocessing (client-side)

- resize source image to 512 × 512, stretching rather than cropping
- convert RGB to `[0, 1]` float
- normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- transpose to NCHW

### Postprocessing

- apply sigmoid (the ONNX file outputs logits, measured around -72..64) to
  get a `[0, 1]` mask
- resize the mask back to the original image dimensions

darktable's one-click masks refine the edges of the upsampled mask on the
image with a guided filter, over a window that widens with the upsampling;
that matters more at 512 than at 1024.

## Quality

Compared on four photos with BiRefNet as the reference:

| Model         | CPU inference | mask on the surroundings | busy scene (bow with rigging) |
| ------------- | ------------- | ------------------------ | ----------------------------- |
| BiRefNet      | ~9.3 s        | reference                | bollard, hull and rigging     |
| BiRefNet Lite | ~4.8 s        | 0.9% of it above 5%      | bollard, hull and rigging     |
| BiRefNet 512  | ~2.4 s        | 0.6% of it above 5%      | the mast and lines only       |

On two trams in a street, a cat and a stuffed toy it gives solid masks
that agree with BiRefNet's (IoU .96 to .99). Its weakness is the busy
scene, where it misses the bollard and most of the hull.

Inference took about 2.5 s per image on CPU in `dtai demo` over the three
`samples/mask` images; the table's figures come from a separate run on the
same CPU.

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | MIT                                                                                                |
| OSAID v1.0               | No: the training data of this checkpoint is not stated                                             |
| MOF                      | Not classified: no data card for this checkpoint                                                   |
| Training data license    | Unknown                                                                                            |
| Training data provenance | Not stated on the model card; the general BiRefNet's card says the DIS models are trained on DIS-TR, the training split of DIS5K |
| Training code            | [MIT](https://github.com/ZhengPeng7/BiRefNet)                                                      |
| Known limitations        | Misses large parts of the subject in busy scenes; edges softer than at 1024 before refinement      |
| Published research       | [BiRefNet](https://arxiv.org/abs/2401.03407) (CAAI AIR 2024)                                       |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic subject / foreground segmentation; no clicks required                                    |
| Reproducibility          | Full pipeline – ONNX is downloaded from the `onnx-community` export of the official checkpoint     |
