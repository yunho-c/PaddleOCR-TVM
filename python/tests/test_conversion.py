from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from paddleocr_tvm.conversion import canonicalize_onnx_model


def test_canonicalize_onnx_model_strips_identity_and_fills_resize_roi(tmp_path: Path) -> None:
    roi_name = "empty_roi"
    identity_output = "shape_passthrough"
    model_path = tmp_path / "model.onnx"

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_raw"], name="Shape.0"),
            helper.make_node("Identity", ["shape_raw"], [identity_output], name="Identity.0"),
            helper.make_node(
                "Constant",
                [],
                [roi_name],
                name="Constant.0",
                value=numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name),
            ),
            helper.make_node(
                "Constant",
                [],
                ["scales"],
                name="Constant.1",
                value=numpy_helper.from_array(
                    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                    name="scales",
                ),
            ),
            helper.make_node(
                "Resize",
                ["x", roi_name, "scales"],
                ["resized"],
                name="Resize.0",
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            ),
            helper.make_node("Slice", [identity_output], ["shape_slice"], name="Slice.0"),
        ],
        name="test_graph",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8]),
        ],
        outputs=[
            helper.make_tensor_value_info("resized", TensorProto.FLOAT, [1, 3, 16, 16]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)

    canonicalize_onnx_model(model_path)

    model = onnx.load(model_path)
    assert all(node.op_type != "Identity" for node in model.graph.node)

    slice_node = next(node for node in model.graph.node if node.name == "Slice.0")
    assert slice_node.input[0] == "shape_raw"

    roi_constant = next(node for node in model.graph.node if node.output[0] == roi_name)
    roi_attr = next(attribute for attribute in roi_constant.attribute if attribute.name == "value")
    roi_value = numpy_helper.to_array(roi_attr.t)
    assert roi_value.shape == (8,)
    assert np.all(roi_value == 0.0)
