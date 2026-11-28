"""Regenerate CLIP text fixtures with uv run --with onnx generate_clip_text.py."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

shape = [1, 77]
for suffix, names in (
    ("ids", ["input_ids"]),
    ("mask", ["input_ids", "attention_mask"]),
    ("unknown", ["input_ids", "attention_mask", "extra_input"]),
    ("missing_ids", ["attention_mask"]),
):
    inputs = [helper.make_tensor_value_info(name, TensorProto.INT64, shape)
              for name in names]
    output = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 512])
    nodes = []
    source = names[0]
    if len(names) > 1:
        source = "combined"
        nodes.append(helper.make_node("Concat", names, [source], axis=1))
    pads = helper.make_tensor("pads", TensorProto.INT64, [4],
                              [0, 0, 0, 512 - shape[1] * len(names)])
    nodes.append(helper.make_node("Cast", [source], ["float_values"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Pad", ["float_values", pads.name], [output.name]))
    graph = helper.make_graph(nodes, f"clip_text_{suffix}", inputs, [output], [pads])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name(f"clip_text_{suffix}.onnx"))
