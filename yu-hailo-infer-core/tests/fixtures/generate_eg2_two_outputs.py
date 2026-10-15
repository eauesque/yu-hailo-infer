"""Regenerate eg2_two_outputs.onnx with ONNX; Rust tests need no Python deps."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

inputs = [helper.make_tensor_value_info(name, TensorProto.INT64, [1, "seq"])
          for name in ("input_ids", "attention_mask")]
inputs += [helper.make_tensor_value_info(name, TensorProto.FLOAT, [0, 512])
           for name in ("image_features", "video_features", "audio_features")]
values = [float(i + 1) for i in range(256)] + [1000.0] * 512
outputs = [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 1, 768]),
           helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [1, 768])]
nodes = [helper.make_node("Constant", [], [out.name],
                         value=helper.make_tensor(out.name + "_value", TensorProto.FLOAT,
                                                  shape, values))
         for out, shape in zip(outputs, ([1, 1, 768], [1, 768]))]
model = helper.make_model(helper.make_graph(nodes, "eg2_two_outputs", inputs, outputs),
                          opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("eg2_two_outputs.onnx"))
