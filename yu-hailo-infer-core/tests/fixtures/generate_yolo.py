"""Regenerate: uv run --no-project --with onnx python generate_yolo.py"""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

values = [0.0] * (84 * 2)
for channel, pair in enumerate(([320, 324], [320, 320], [320, 320], [160, 160], [0.9, 0.8])):
    values[channel * 2:channel * 2 + 2] = pair
output = helper.make_tensor("detections", TensorProto.FLOAT, [1, 84, 2], values)
graph = helper.make_graph(
    [helper.make_node("Constant", [], ["output"], value=output)],
    "tiny-yolo",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 2])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("yolo_constant.onnx"))
