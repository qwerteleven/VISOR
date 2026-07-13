import numpy as np
import pytest

from utils.cuda_handler import allocate_buffers, free_buffers, get_input_output

import tests.unit.utils.cuda.tensorrt as trt


class FakeEngine:
    def __init__(self, tensors, has_implicit_batch_dimension=False, max_batch_size=1):
        self._tensors = tensors
        self.has_implicit_batch_dimension = has_implicit_batch_dimension
        self.max_batch_size = max_batch_size

    @property
    def num_io_tensors(self):
        return len(self._tensors)

    def get_tensor_name(self, i):
        return self._tensors[i]["name"]

    def get_tensor_shape(self, name):
        return next(t["shape"] for t in self._tensors if t["name"] == name)

    def get_tensor_profile_shape(self, name, profile_idx):
        shape = next(t["shape"] for t in self._tensors if t["name"] == name)
        return [shape, shape, shape]

    def get_tensor_dtype(self, name):
        return next(t["dtype"] for t in self._tensors if t["name"] == name)

    def get_tensor_mode(self, name):
        return next(t["mode"] for t in self._tensors if t["name"] == name)


def make_simple_engine():
    return FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (1, 3),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
            {
                "name": "output0",
                "shape": (1, 2),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.OUTPUT,
            },
        ]
    )


def test_allocate_buffers_classifies_inputs_and_outputs_correctly():
    engine = make_simple_engine()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert inputs[0].host.shape == (3,)
    assert outputs[0].host.shape == (2,)

    free_buffers(inputs, outputs, stream)


def test_allocate_buffers_bindings_match_device_pointers_in_order():
    engine = make_simple_engine()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    assert bindings == [inputs[0].device, outputs[0].device]

    free_buffers(inputs, outputs, stream)


def test_allocate_buffers_raises_on_dynamic_shape_without_profile_idx():
    engine = FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (-1, 3),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
        ]
    )

    with pytest.raises(ValueError, match="dynamic dimension"):
        allocate_buffers(engine)


def test_allocate_buffers_resolves_dynamic_shape_via_profile_idx():
    engine = FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (2, 4),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
        ]
    )

    inputs, outputs, bindings, stream = allocate_buffers(engine, profile_idx=0)

    assert inputs[0].host.shape == (8,)  # volume of (2, 4)
    free_buffers(inputs, outputs, stream)


def test_allocate_buffers_uses_context_shape_when_context_given():
    engine = FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (100, 100),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
        ]
    )

    class FakeContextWithShape:
        def get_tensor_shape(self, name):
            return (2, 2)

    inputs, outputs, bindings, stream = allocate_buffers(
        engine, context=FakeContextWithShape()
    )

    assert inputs[0].host.shape == (4,)
    free_buffers(inputs, outputs, stream)


def test_allocate_buffers_applies_implicit_batch_multiplier():
    engine = FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (4,),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
        ],
        has_implicit_batch_dimension=True,
        max_batch_size=3,
    )

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    assert inputs[0].host.shape == (12,)  # 4 * 3
    free_buffers(inputs, outputs, stream)


def test_allocate_buffers_uses_ml_dtypes_fallback_for_bf16_tensors():
    engine = FakeEngine(
        tensors=[
            {
                "name": "input0",
                "shape": (4,),
                "dtype": trt.DataType.BF16,
                "mode": trt.TensorIOMode.INPUT,
            },
        ]
    )

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    import ml_dtypes

    assert inputs[0].host.dtype == np.dtype(ml_dtypes.bfloat16)
    free_buffers(inputs, outputs, stream)


def test_get_input_output_indexes_tensors_separately():
    engine = FakeEngine(
        tensors=[
            {
                "name": "in_a",
                "shape": (1,),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
            {
                "name": "out_a",
                "shape": (1,),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.OUTPUT,
            },
            {
                "name": "in_b",
                "shape": (1,),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.INPUT,
            },
            {
                "name": "out_b",
                "shape": (1,),
                "dtype": trt.DataType.FLOAT,
                "mode": trt.TensorIOMode.OUTPUT,
            },
        ]
    )

    inputs, outputs = get_input_output(engine)

    assert inputs == {"in_a": 0, "in_b": 1}
    assert outputs == {"out_a": 0, "out_b": 1}
