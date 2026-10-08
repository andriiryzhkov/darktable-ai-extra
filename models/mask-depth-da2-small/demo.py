"""Run Depth Anything V2 Small on a sample image and save the depth map.

The graph's spatial size is dynamic, so this follows upstream's
preprocessing rather than stretching to a square: the aspect ratio is
kept, the short side brought to 518 and both sides to multiples of 14,
the ViT patch size.

I/O contract, measured from the graph rather than taken on trust:
  pixel_values     float32 [1,3,H,W], RGB in [0,1] then ImageNet-
                   normalized, H and W multiples of 14
  predicted_depth  float32 [1,H,W] relative inverse depth: larger is
                   nearer, unbounded, normalized per image here

Writes the depth map as an 8-bit grayscale PNG (bright is near), plus a
`-compare` image putting the source and the map side by side.

Usage:
  python3 models/mask-depth-da2-small/demo.py \
      --model output/mask-depth-da2-small/model.onnx \
      --image samples/mask-depth/example_01.jpg \
      --output output/mask-depth-da2-small-demo/example_01.png
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
    "input_sizes": [518],
    "ensure_multiple_of": 14,
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
}


def _load_attributes(model_path):
    """Read attributes from config.json next to the model, else defaults."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    attrs = dict(DEFAULTS)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            attrs.update(json.load(f).get("attributes", {}))
    return attrs


def input_size(width, height, short, multiple):
    """The short side at `short`, the aspect kept, both sides multiples."""
    scale = short / min(width, height)
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def preprocess(image, size, mean, std):
    """RGB image to a normalized NCHW batch of one, at size (W, H)."""
    arr = np.asarray(image.resize(size, Image.BICUBIC), np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def normalize(depth):
    """Per image to [0,1]: the output has no fixed scale."""
    lo, hi = float(depth.min()), float(depth.max())
    return (depth - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(depth)


def save_comparison(image, depth, path):
    """source | depth map, for eyeballing the result."""
    gray = Image.fromarray((depth * 255).astype(np.uint8)).convert("RGB")
    gap = np.full((image.size[1], 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([np.asarray(image), gap, np.asarray(gray)], axis=1)).save(path)


def run_inference(model, image, output):
    attrs = _load_attributes(model)
    short = int(attrs["input_sizes"][0])
    multiple = int(attrs["ensure_multiple_of"])
    mean = np.asarray(attrs["input_mean"], np.float32)
    std = np.asarray(attrs["input_std"], np.float32)

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
    size = input_size(img.size[0], img.size[1], short, multiple)

    t = time.time()
    raw = session.run(None, {input_name: preprocess(img, size, mean, std)})[0]
    print(f"    {img.size[0]}x{img.size[1]} at {size[0]}x{size[1]} in {time.time() - t:.2f}s")

    # (1,H,W) -> (H,W), then back onto the source image's pixel grid
    small = Image.fromarray(normalize(raw[0].astype(np.float32)))
    depth = np.asarray(small.resize(img.size, Image.BICUBIC), np.float32).clip(0.0, 1.0)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    Image.fromarray((depth * 255).astype(np.uint8)).save(output)
    print(f"    wrote {output}")

    root, ext = os.path.splitext(output)
    save_comparison(img, depth, f"{root}-compare{ext}")
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
