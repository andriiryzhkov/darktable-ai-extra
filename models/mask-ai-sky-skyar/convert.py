"""Export SkyAR's sky matting network to ONNX.

The network is ResNet50FCN with CoordConv (`coord_resnet50`), defined in
vendor/SkyAR/networks.py. Three things stand between that file and a
plain export:

  - networks.py imports SkyAR's `utils` and matplotlib at the top without
    using either for the network. `utils` imports `compare_ssim` from
    scikit-image, which no longer has it, so both are stubbed out here
    rather than installed
  - the constructor calls `models.resnet50(pretrained=True)`, which
    downloads ImageNet weights only for the checkpoint to overwrite them;
    it is pointed at an empty ResNet-50 instead
  - the checkpoint is a zip from Google Drive whose best_ckpt.pt also
    carries the optimizer state; only `model_G_state_dict` is loaded

The export is traced at the fixed input size and checked against PyTorch.
"""

import os
import sys
import types
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

CKPT_IN_ZIP = "checkpoints_G_coord_resnet50/best_ckpt.pt"


def _build_network(repo_dir):
    for name in ("utils", "matplotlib", "matplotlib.pyplot"):
        sys.modules.setdefault(name, types.ModuleType(name))

    resnet50 = torchvision.models.resnet50
    torchvision.models.resnet50 = lambda *args, **kwargs: resnet50(weights=None)
    try:
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        import networks
        return networks.ResNet50FCN(coordconv=True)
    finally:
        torchvision.models.resnet50 = resnet50


def convert(checkpoint, repo_dir, output_dir, size=384, opset=17):
    size = int(size)
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "model.onnx")

    net = _build_network(repo_dir)
    with zipfile.ZipFile(checkpoint) as zf, zf.open(CKPT_IN_ZIP) as f:
        state = torch.load(f, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_G_state_dict"])
    net.eval()
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  coord_resnet50 from epoch {state.get('best_epoch_id')}, {params:.1f}M parameters")

    x = torch.rand(1, 3, size, size)
    torch.onnx.export(net, x, target,
                      input_names=["input_image"], output_names=["output_image"],
                      opset_version=int(opset), do_constant_folding=True,
                      dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    print(f"  wrote {target} ({size}x{size}, opset {opset})")

    with torch.no_grad():
        ref = net(x).numpy()
    session = ort.InferenceSession(target, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input_image": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    print(f"  ONNX matches PyTorch: max diff {diff:.2e}")
    if diff > 1e-4:
        raise ValueError("the export does not reproduce the PyTorch output")
