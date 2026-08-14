# LaMa

Large-mask inpainting. Fills a user-marked region with plausible content –
removing power lines, sensor dust, or unwanted objects. Fast Fourier
Convolutions give the network a global receptive field from the first layer,
which is why it reconstructs large contiguous holes and repetitive structure
(brickwork, fences, foliage) far better than patch-based healing.

Also usable for outpainting: the transparent wedges a rotate or perspective
correction leaves at the frame edges are just another mask.

## Source

- Repository: <https://github.com/advimman/lama>
- Paper: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022)
- License: Apache-2.0 (Samsung Research)
- ONNX weights: [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`, Apache-2.0, exported from the official `big-lama` checkpoint)

Upstream ships PyTorch checkpoints, not ONNX. Exporting them requires patching
the `FourierUnit` so `torch.fft.rfftn` / `irfftn` trace into ops onnxruntime
supports. Carve published that work, so this package downloads the export
directly and runs no local conversion.

## Architecture

ResNet-style generator whose residual blocks are Fast Fourier Convolutions.
Each FFC block splits its channels into a local branch (ordinary convolution)
and a global branch that operates on the real FFT of the feature map, so
information crosses the whole image in one layer instead of accumulating
through depth. That is what makes the receptive field resolution-robust and
lets a hole be filled from context far outside it.

## ONNX Model

| Direction | Tensor | Shape                    | Type    |
| --------- | ------ | ------------------------ | ------- |
| in        | image  | B x 3 x 512 x 512        | float32 |
| in        | mask   | B x 1 x 512 x 512        | float32 |
| out       | output | B x 3 x 512 x 512        | float32 |

opset 17, ~198 MiB. **Only the batch axis is dynamic** – the spatial dims are
baked in, and the graph rejects any other size outright rather than
interpolating. Batching gives no throughput win (measured slightly worse per
tile), so process one window at a time.

The values below were measured against this graph, not taken from upstream
documentation, and two of them differ from what you would assume.

### Preprocessing (client-side)

- RGB in `[0, 1]`, NCHW float32 – no mean/std normalisation
- mask float32, `1` marks pixels to fill, `0` marks pixels to keep
- **the hole needs no pre-filling.** The graph zeroes the masked region
  itself; changing the pixels under the mask does not change the output at all
- whatever sits under a transparent alpha export is usually black, and that
  black must not be allowed to bleed across the mask edge when resampling –
  neutralise the hole first, or the fill matches a darkened neighbourhood

### Postprocessing

- **the output is in `[0, 255]`, not `[0, 1]`** – divide by 255
- **the result is already composited.** Unmasked pixels come back bit-exact,
  so no feathering or blending is needed; take the masked pixels and paste

### Applying it to a full-resolution photo

A 512 window spans roughly a tenth of a 26 MP frame, so `demo.py` plans
windows per connected region of the mask:

- a blob that fits gets **one** window sized to leave context around it,
  resampled down if the blob is larger than 512. One window per blob matters:
  splitting a compact blob across several produces mismatched fills meeting
  along straight seams
- a region too long to enclose – an edge wedge, a power line – is **tiled**
  along its length instead, which keeps most windows at native resolution

`attributes.max_hole_fraction` (0.4) is the cutoff: past that share of a
window, LaMa has too little context left and the window grows.

## Selection Criteria

| Property                 | Value                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                       |
| OSAID v1.0               | Open Weights                                                                                     |
| MOF                      | Class II (Open Tooling)                                                                          |
| Training data license    | [Places2](http://places2.csail.mit.edu/): non-commercial research and educational use only; copyright in the individual photographs remains with their owners |
| Training data provenance | [Places365-Challenge](http://places2.csail.mit.edu/) (~8M scene photographs, MIT CSAIL) with synthetic large-mask augmentation generated at training time |
| Training code            | [Apache-2.0](https://github.com/advimman/lama)                                                   |
| Known limitations        | Places2 is research-only, so the permissive weight license does not carry through to commercial use of the output. The export is fixed at 512x512: a hole wider than ~205 px needs a resampled window and comes back softer, and a hole over 512 px cannot be filled in one pass at native resolution. Tiled windows along long masks can disagree at their seams. darktable has no built-in `inpaint` task, so the model is driven from Lua |
| Published research       | [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022) |
| Inference                | Local only, no cloud dependencies                                                                |
| Scope                    | Large-mask inpainting and object removal; edge fill after rotate / perspective                   |
| Reproducibility          | Download only – the ONNX is fetched pre-exported from Carve; there is no local conversion step to reproduce |

## Using it from darktable

`task: inpaint` is not one of the tasks darktable's C code dispatches on
(`denoise`, `rawdenoise`, `upscale`, `mask`), so installing this model adds no
UI. It is reachable through the Lua API, which takes any task string:

```lua
local id = darktable.ai.model_for_task("inpaint")
local ctx = darktable.ai.load_model(id)
-- build image and mask tensors, ctx:run(...), divide the result by 255
```

## Demo

```console
dtai demo inpaint-lama
```

Runs `demo.py` over `samples/inpaint/`. The mask comes from each sample's own
**alpha channel** – transparent means fill – so a sample is one file with no
sidecar. Results land in `output/inpaint-lama-demo/`, each next to a
`-compare` image putting the input and result side by side.
