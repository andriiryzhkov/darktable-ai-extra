"""Run LaMa on a sample image and save the inpainted result.

The ONNX graph has a hard 512x512 input, so a full-resolution photo is
handled with the `crop_around_mask` strategy this model declares: cut a 512
window around the masked region at native resolution, inpaint that, paste it
back. When the hole is too big for a window to leave usable context, fall
back to downscaling the whole image – which fills it, but visibly softer.

The masked region comes from a JSON sidecar next to the sample image, or
from `demo.image_args` in model.yaml. Either supply rectangles in
normalized coordinates:

    {"rects": ["0.30,0.62,0.10,0.08"]}

or point at a mask image, relative to the sample, where white = inpaint:

    {"mask": "example_01_mask.png"}

I/O contract, measured from the graph rather than taken on trust:
  image  float32 [1,3,512,512] RGB in [0,1]
  mask   float32 [1,1,512,512], 1 marks the hole; the graph zeroes the
         masked pixels itself, so the hole needs no pre-filling
  output float32 [1,3,512,512] in [0,255] – not [0,1] – and already
         composited with the unmasked input, so no feathering is needed

Usage:
  python3 models/inpaint-lama/demo.py \
      --model output/inpaint-lama/model.onnx \
      --image samples/inpaint/example_01.jpg \
      --output output/inpaint-lama-demo/example_01.png \
      --rects 0.30,0.62,0.10,0.08
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from scipy import ndimage
from PIL import Image, ImageFilter

# Used only when config.json is absent, so the script still works against a
# raw download of the ONNX.
DEFAULTS = {
    "input_sizes": [512],
    "max_hole_fraction": 0.4,
    "output_scale": 255,
}


def _load_attributes(model_path):
    """Read attributes from config.json next to the model, else defaults."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    attrs = dict(DEFAULTS)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            attrs.update(json.load(f).get("attributes", {}))
    return attrs


def _parse_rects(rects, width, height):
    """Normalized 'x,y,w,h' strings (or 4-tuples) to pixel rectangles."""
    out = []
    for r in rects:
        parts = [float(p) for p in r.split(",")] if isinstance(r, str) else list(r)
        if len(parts) != 4:
            raise ValueError(f"rect needs four values x,y,w,h – got {r!r}")
        x, y, w, h = parts
        # values <= 1 are fractions of the image, above that they're pixels
        if max(parts) <= 1.0:
            x, y, w, h = x * width, y * height, w * width, h * height
        out.append((int(x), int(y), max(1, int(w)), max(1, int(h))))
    return out


def black_border_mask(image, threshold=12, grow=3):
    """Mask the black wedges rotate/perspective leaves at the frame edges.

    Those pixels are exactly zero before JPEG, so a low threshold finds
    them. Only regions connected to an edge count – a genuinely black
    subject in the middle of the frame must not be treated as missing.
    `grow` widens the mask a little to swallow the compression-softened
    boundary, which would otherwise smear dark fringing into the fill.
    """
    dark = image.max(axis=2) <= threshold
    if not dark.any():
        return np.zeros(image.shape[:2], np.float32)

    # keep only dark pixels reachable from the border, row- and column-wise
    keep = np.zeros_like(dark)
    for axis in (0, 1):
        d = dark if axis == 0 else dark.T
        k = np.zeros_like(d)
        # a run of dark pixels touching either end of the line
        lead = np.logical_and.accumulate(d, axis=0)
        trail = np.logical_and.accumulate(d[::-1], axis=0)[::-1]
        k |= lead | trail
        keep |= k if axis == 0 else k.T

    mask = (dark & keep).astype(np.uint8) * 255
    if grow > 0:
        m = Image.fromarray(mask).filter(ImageFilter.MaxFilter(2 * grow + 1))
        mask = np.asarray(m)
    return (mask > 127).astype(np.float32)


def alpha_mask(path, threshold=128):
    """Mask from a sample's own alpha channel – transparent means 'fill'.

    The tidiest way to hand a region to the model: one file, no sidecar.
    Paint the areas to remove as transparent in any editor and save as
    PNG/TIFF/WebP. It also matches how a rotate/perspective export would
    naturally mark the corners it could not fill.

    Returns None when the file has no usable alpha, so callers can fall
    back to the other ways of specifying a region.
    """
    im = Image.open(path)
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        return None
    alpha = np.asarray(im.convert("RGBA"))[..., 3]
    mask = (alpha < threshold).astype(np.float32)
    return mask if mask.any() else None


def build_mask(shape, rects=None, mask_path=None, auto=None, image=None):
    """HW float32 mask, 1 = inpaint, from rectangles, an image, or auto."""
    height, width = shape
    if auto:
        if auto != "black_borders":
            raise ValueError(f"unknown auto mask mode: {auto!r}")
        mask = black_border_mask(image)
    elif mask_path:
        m = Image.open(mask_path).convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        mask = (np.asarray(m, np.float32) / 255.0 > 0.5).astype(np.float32)
    else:
        mask = np.zeros((height, width), np.float32)
        for x, y, w, h in _parse_rects(rects or [], width, height):
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0

    if not mask.any():
        raise ValueError("mask is empty – nothing to inpaint")
    return mask


def _pad_to(arr, tile, fill=0):
    """Pad an HW or HWC array up to tile x tile (images smaller than a tile)."""
    h, w = arr.shape[:2]
    if h == tile and w == tile:
        return arr
    out = np.full((tile, tile) + arr.shape[2:], fill, arr.dtype)
    out[:h, :w] = arr
    return out


def _forward(session, tile_rgb, tile_mask, output_scale):
    """One forward pass: uint8 HWC in, uint8 HWC out."""
    img = tile_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    msk = tile_mask.astype(np.float32)[None, None]
    out = session.run(None, {"image": img, "mask": msk})[0]
    out = out[0].transpose(1, 2, 0) * (255.0 / output_scale)
    return np.clip(out, 0, 255).astype(np.uint8)


def plan_windows(mask, tile, fraction):
    """Cover the masked pixels with windows, each leaving enough context.

    A single window per region is wrong for anything long and thin – an
    edge wedge or a power line has a bounding box spanning the frame, but
    is only tens of pixels thick. Tiling along it keeps every window at
    native resolution instead of resampling the whole frame down.

    Yields (x, y, side). `side` exceeds `tile` only when the hole is too
    fat for a native window, in which case the caller resamples.
    """
    height, width = mask.shape
    limit = min(width, height)

    labels, count = ndimage.label(mask > 0.5)
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bw, bh = x1 - x0, y1 - y0

        # One window for the whole blob whenever it fits. Splitting a
        # compact blob across windows is what produces mismatched patches:
        # each window invents its own fill and they meet along a straight
        # seam. One window resampled is worse in sharpness but coherent,
        # and for a bird against sky that trade is obviously right.
        side = int(min(max(tile, round(max(bw, bh) / fraction)), limit))
        if max(bw, bh) / fraction <= limit:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            ax = int(np.clip(cx - side // 2, 0, max(0, width - side)))
            ay = int(np.clip(cy - side // 2, 0, max(0, height - side)))
            yield ax, ay, side
            continue

        # Too long to enclose – a wedge or a wire. Tile along it instead,
        # which keeps most windows native rather than resampling the frame.
        step = max(1, int(tile * 0.75))
        seen = set()
        for y in range(y0, y1, step):
            for x in range(x0, x1, step):
                sx = int(np.clip(x, 0, max(0, width - tile)))
                sy = int(np.clip(y, 0, max(0, height - tile)))
                if not (labels[sy:sy + tile, sx:sx + tile] == index).any():
                    continue
                w_side, cx, cy = tile, sx + tile // 2, sy + tile // 2
                while w_side < limit:
                    bx = int(np.clip(cx - w_side // 2, 0, max(0, width - w_side)))
                    by = int(np.clip(cy - w_side // 2, 0, max(0, height - w_side)))
                    if mask[by:by + w_side, bx:bx + w_side].mean() <= fraction:
                        break
                    w_side = int(min(w_side * 1.6, limit))
                bx = int(np.clip(cx - w_side // 2, 0, max(0, width - w_side)))
                by = int(np.clip(cy - w_side // 2, 0, max(0, height - w_side)))
                if (bx, by, w_side) not in seen:
                    seen.add((bx, by, w_side))
                    yield bx, by, w_side


def inpaint_region(session, image, mask, attrs):
    """Inpaint one region of `mask` (HW, 1=hole) in `image` (HWC uint8).

    One window per region, sized so the hole occupies at most
    `max_hole_fraction` of it – LaMa needs the rest as context. If that
    window is bigger than the model's fixed input it is resampled down to
    fit and the result lifted back up, which costs sharpness in the filled
    pixels only. On a 26MP file this is the normal path: a 512 window spans
    barely a tenth of the frame, so almost any real subject needs one.

    Shrinking a *window* rather than the whole frame is what keeps this
    usable – a full-frame 4160x6240 -> 512 reduction would throw away 8x
    more detail than cropping first.
    """
    tile = int(attrs["input_sizes"][0])
    scale = float(attrs.get("output_scale", 255))
    fraction = float(attrs.get("max_hole_fraction", 0.4))
    height, width = mask.shape

    windows = list(plan_windows(mask, tile, fraction))
    if not windows:
        return image

    native = sum(1 for _, _, s in windows if s == tile)
    grown = [s for _, _, s in windows if s != tile]
    note = f"{native} native"
    if grown:
        note += f", {len(grown)} resampled (up to {max(grown)}px -> {tile})"
    print(f"    {len(windows)} window(s): {note}")

    out = image.copy()
    for x, y, side in windows:
        w, h = min(side, width - x), min(side, height - y)
        # Read context from the ORIGINAL frame but write into `out`, so
        # windows don't feed each other's output back in as if it were
        # real detail. Overlaps then agree rather than compounding.
        crop = _pad_to(image[y:y + h, x:x + w], side)
        m_crop = _pad_to(mask[y:y + h, x:x + w], side)

        if side == tile:
            filled = _forward(session, crop, m_crop, scale)
        else:
            # Neutralise the hole before resampling. Whatever sits under
            # the mask – black from an alpha export, or old content – would
            # otherwise bleed across the mask edge during downsampling and
            # darken the very context LaMa matches its fill to.
            sel_full = m_crop > 0.5
            neutral = crop.copy()
            if (~sel_full).any():
                neutral[sel_full] = crop[~sel_full].mean(axis=0).astype(crop.dtype)

            src = np.asarray(Image.fromarray(neutral).resize((tile, tile),
                                                             Image.LANCZOS))
            # Grow the downsampled mask by a pixel: BOX averaging spreads
            # the hole slightly, and any masked pixel left unmarked would
            # be treated as real content to preserve.
            m_small = np.asarray(
                Image.fromarray((m_crop * 255).astype(np.uint8))
                .resize((tile, tile), Image.BOX)
                .filter(ImageFilter.MaxFilter(3)), np.float32) / 255.0
            m_small = (m_small > 0.05).astype(np.float32)
            small = _forward(session, src, m_small, scale)
            filled = np.asarray(Image.fromarray(small).resize((side, side),
                                                              Image.LANCZOS))

        # Only masked pixels are taken. Everywhere else the round-trip
        # through the model (and any resample) would soften pixels that
        # were already correct.
        region = out[y:y + h, x:x + w]
        sel = m_crop[:h, :w] > 0.5
        region[sel] = filled[:h, :w][sel]
        out[y:y + h, x:x + w] = region

    return out


def inpaint(session, image, mask, attrs, regions=None):
    """Inpaint every region in turn. `regions` are separate HW masks.

    Disjoint holes get their own window each – a shared bounding box around
    two subjects at opposite corners would span the whole frame and force a
    needlessly coarse resample.
    """
    out = image
    for i, region in enumerate(regions or [mask], 1):
        if not region.any():
            continue
        if regions and len(regions) > 1:
            print(f"    region {i}/{len(regions)}")
        out = inpaint_region(session, out, region, attrs)
    return out


def save_comparison(before, after, mask, path):
    """before (hole tinted red) | after, for eyeballing the result."""
    marked = before.copy()
    sel = mask > 0.5
    marked[sel] = (0.5 * marked[sel] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    gap = np.full((before.shape[0], 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([marked, gap, after], axis=1)).save(path)


def run_inference(model, image, output, rects=None, mask=None, auto=None):
    attrs = _load_attributes(model)
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    img = np.asarray(Image.open(image).convert("RGB"))
    # a mask path in the sidecar is relative to the sample image
    mask_path = os.path.join(os.path.dirname(image), mask) if mask else None

    if rects or mask_path or auto:
        msk = build_mask(img.shape[:2], rects=rects, mask_path=mask_path,
                         auto=auto, image=img)
        source = "sidecar"
    else:
        # Nothing specified: the sample's own transparency is the mask.
        msk = alpha_mask(image)
        source = "alpha channel"
        if msk is None:
            print("    no region to fill – save the sample as a PNG with the "
                  "areas to remove made transparent; skipping")
            return
    print(f"    mask from {source}: {msk.mean():.1%} of the frame")

    t = time.time()
    result = inpaint(session, img, msk, attrs)
    print(f"    {img.shape[1]}x{img.shape[0]} in {time.time() - t:.2f}s")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    Image.fromarray(result).save(output)
    print(f"    wrote {output}")

    root, ext = os.path.splitext(output)
    save_comparison(img, result, msk, f"{root}-compare{ext}")
    print(f"    wrote {root}-compare{ext}")


def demo(model, image, output, **kwargs):
    """Pipeline entry point."""
    run_inference(model, image, output,
                  rects=kwargs.get("rects"), mask=kwargs.get("mask"),
                  auto=kwargs.get("auto"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rects", nargs="*", default=None,
                        help="x,y,w,h – fractions of the image, or pixels")
    parser.add_argument("--mask", default=None,
                        help="mask image (white = inpaint), relative to --image")
    parser.add_argument("--auto", default=None, choices=["black_borders"],
                        help="derive the mask automatically")
    args = parser.parse_args()
    demo(args.model, args.image, args.output,
         rects=args.rects, mask=args.mask, auto=args.auto)


if __name__ == "__main__":
    main()
