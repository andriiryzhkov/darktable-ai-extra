"""Cut the isnet-general-use ONNX down to its prediction.

IS-Net returns twelve tensors: six sigmoid side maps, finest first, and the
six decoder feature maps they come from, which only matter for deep
supervision during training. darktable's AI mask runner takes models with
exactly one output, so this keeps the first side map, the one upstream's
Inference.py uses as the result, and drops the rest. The weights are
untouched.

The prediction is IS-Net's `side1` head: a 3x3 convolution from 64
channels to one, upsampled to the input size, then the sigmoid. The check
below fails the conversion if a different export orders the outputs
otherwise.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort

IN_NAME = "input_image"
OUT_NAME = "output_image"
SIZE = 1024


def _check_prediction(graph, name):
    """The output must be Sigmoid(Resize(Conv)) with a 64-to-1 kernel."""
    producer = {o: n for n in graph.node for o in n.output}
    weights = {i.name: i for i in graph.initializer}
    sig = producer.get(name)
    resize = producer.get(sig.input[0]) if sig is not None else None
    conv = producer.get(resize.input[0]) if resize is not None else None
    if sig is None or sig.op_type != "Sigmoid" or resize is None \
       or resize.op_type != "Resize" or conv is None or conv.op_type != "Conv":
        raise ValueError(f"{name} is not a sigmoid over an upsampled convolution")
    w = weights.get(conv.input[1])
    if w is None or list(w.dims)[:2] != [1, 64]:
        raise ValueError(f"{name} is not the side1 head, 64 channels to one")


def _keep_first_output(model):
    graph = model.graph
    old_in = graph.input[0].name
    old_out = graph.output[0].name
    _check_prediction(graph, old_out)

    del graph.output[1:]

    # drop what only fed the removed outputs, so the graph has no dead
    # nodes for onnxruntime to warn about
    live = {old_out}
    kept = []
    for node in reversed(graph.node):
        if any(o in live for o in node.output):
            kept.append(node)
            live.update(i for i in node.input if i)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    used = {i for n in graph.node for i in n.input}
    inits = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)

    for node in graph.node:
        node.input[:] = [IN_NAME if i == old_in else i for i in node.input]
        node.output[:] = [OUT_NAME if o == old_out else o for o in node.output]
    graph.input[0].name = IN_NAME
    graph.output[0].name = OUT_NAME
    return old_in, old_out


def convert(source, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "model.onnx")

    model = onnx.load(source)
    print(f"  {source}: {len(model.graph.output)} outputs")
    old_in, old_out = _keep_first_output(model)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    print(f"  wrote {target} ({old_in} -> {IN_NAME}, {old_out} -> {OUT_NAME})")

    # the cut graph runs the same nodes, so it must give the same prediction
    x = np.random.default_rng(0).standard_normal((1, 3, SIZE, SIZE), np.float32)
    ref = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
    cut = ort.InferenceSession(target, providers=["CPUExecutionProvider"])
    a = ref.run([old_out], {old_in: x})[0]
    b = cut.run(None, {IN_NAME: x})[0]
    diff = float(np.abs(a - b).max())
    print(f"  prediction matches the original: max diff {diff:.2e}")
    if diff > 1e-6:
        raise ValueError("the cut graph does not reproduce the prediction")
