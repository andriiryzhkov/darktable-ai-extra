"""Run BiRefNet Lite on a sample image and save the subject mask.

One forward pass over the whole frame resampled to the model's fixed
1024x1024 input – there are no click prompts and nothing to tile, so this
is the `resize_whole_image` strategy the model declares.

I/O contract, measured from the graph rather than taken on trust:
  input_image   float32 [1,3,1024,1024], RGB in [0,1] then ImageNet-
                normalized. The fp16 package casts internally and still
                takes float32 at the boundary
  output_image  float32 [1,1,1024,1024] logits, around -44..25. Sigmoid
                gives a soft alpha, not a binary mask

Writes the mask as an 8-bit grayscale PNG, plus a `-compare` image putting
the source, the mask and a red overlay side by side.

Usage:
  python3 models/mask-subject-birefnet-lite/demo.py \
      --model output/mask-subject-birefnet-lite/model.onnx \
      --image samples/mask/example_01.jpg \
      --output output/mask-subject-birefnet-lite-demo/example_01.png
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# Used only when config.json is absent, so the script still works against a
# raw download of the ONNX.
DEFAULTS = {
    "input_sizes": [1024],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "output_logits": True,
}


def _load_attributes(model_path):
    """Read attributes from config.json next to the model, else defaults."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    attrs = dict(DEFAULTS)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            attrs.update(json.load(f).get("attributes", {}))
    return attrs


def preprocess(image, size, mean, std, ort_type):
    """RGB image to a normalized NCHW batch of one."""
    arr = np.asarray(image.resize((size, size), Image.LANCZOS), np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    arr = (arr - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    # The published fp16 export still declares float32 inputs, but a
    # re-export with fp16 at the boundary would not, so follow the graph.
    dtype = np.float16 if ort_type == "tensor(float16)" else np.float32
    return arr.astype(dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def overlay(image, mask, alpha=0.45):
    """Tint the masked subject red, for eyeballing the boundary."""
    arr = np.asarray(image, np.float32)
    m = mask[:, :, np.newaxis] * alpha
    arr = arr * (1 - m) + np.array([255.0, 0.0, 0.0]) * m
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def save_comparison(image, mask, tinted, path):
    """source | mask | overlay, for eyeballing the result."""
    gray = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    gap = np.full((image.size[1], 8, 3), 255, np.uint8)
    panels = [np.asarray(image), gap, np.asarray(gray), gap, np.asarray(tinted)]
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def run_inference(model, image, output):
    attrs = _load_attributes(model)
    size = int(attrs["input_sizes"][0])
    mean = np.asarray(attrs["input_mean"], np.float32)
    std = np.asarray(attrs["input_std"], np.float32)

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")

    t = time.time()
    raw = session.run(None, {input_meta.name:
                             preprocess(img, size, mean, std, input_meta.type)})[0]
    print(f"    {img.size[0]}x{img.size[1]} in {time.time() - t:.2f}s")

    raw = raw.astype(np.float32)
    # Guard rather than trust: a re-export that bakes the sigmoid in would
    # already be in [0,1], and squashing it twice flattens the boundary.
    if attrs.get("output_logits", True) and (raw.min() < 0.0 or raw.max() > 1.0):
        raw = sigmoid(raw)

    # (1,1,H,W) -> (H,W), then back onto the source image's pixel grid
    small = Image.fromarray((raw[0, 0] * 255).clip(0, 255).astype(np.uint8))
    mask = np.asarray(small.resize(img.size, Image.LANCZOS), np.float32) / 255.0
    print(f"    subject covers {mask.mean():.1%} of the frame")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output)
    print(f"    wrote {output}")

    root, ext = os.path.splitext(output)
    save_comparison(img, mask, overlay(img, mask), f"{root}-compare{ext}")
    print(f"    wrote {root}-compare{ext}")


def demo(model, image, output, **kwargs):
    """Pipeline entry point."""
    run_inference(model, image, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    demo(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
