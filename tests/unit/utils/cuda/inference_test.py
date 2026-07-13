import ml_dtypes
import numpy as np

from utils.cuda_handler import HostDeviceMem, do_inference, to_host_bytes


class FakeContext:
    def __init__(self):
        self.tensor_addresses = {}
        self.execute_calls = []

    def set_tensor_address(self, name, address):
        self.tensor_addresses[name] = address

    def execute_async_v3(self, stream_handle):
        self.execute_calls.append(stream_handle)


class FakeEngine:
    def __init__(self, names):
        self._names = names

    @property
    def num_io_tensors(self):
        return len(self._names)

    def get_tensor_name(self, i):
        return self._names[i]


def test_do_inference_sets_tensor_addresses_for_all_bindings():
    engine = FakeEngine(names=["input0", "output0"])
    context = FakeContext()
    bindings = [111, 222]
    inp = HostDeviceMem(size=2, dtype=np.float32)
    out = HostDeviceMem(size=2, dtype=np.float32)

    do_inference(context, engine, bindings, [inp], [out], stream=1)

    assert context.tensor_addresses == {"input0": 111, "output0": 222}
    inp.free()
    out.free()


def test_do_inference_calls_execute_with_given_stream():
    engine = FakeEngine(names=["input0"])
    context = FakeContext()
    inp = HostDeviceMem(size=1, dtype=np.float32)

    do_inference(context, engine, [inp.device], [inp], [], stream=99)

    assert context.execute_calls == [99]
    inp.free()


def test_do_inference_returns_output_host_arrays():
    engine = FakeEngine(names=["output0"])
    context = FakeContext()
    out = HostDeviceMem(size=3, dtype=np.float32)
    out.host[:] = [7.0, 8.0, 9.0]

    result = do_inference(context, engine, [out.device], [], [out], stream=1)

    assert len(result) == 1
    np.testing.assert_allclose(result[0], [7.0, 8.0, 9.0])
    out.free()


def test_to_host_bytes_ravels_normal_float_array():
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = to_host_bytes(z)

    assert result.shape == (4,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_to_host_bytes_converts_bfloat16_to_uint16_view():
    z = np.array([1.0, 2.0, 3.0], dtype=ml_dtypes.bfloat16)

    result = to_host_bytes(z)

    assert result.dtype == np.uint16
    assert result.shape == (3,)


def test_to_host_bytes_bfloat16_view_is_bit_accurate_not_value_converted():
    z = np.array([1.5], dtype=ml_dtypes.bfloat16)

    result = to_host_bytes(z)
    roundtrip = result.view(ml_dtypes.bfloat16)

    assert roundtrip[0] == np.array(1.5, dtype=ml_dtypes.bfloat16)


def test_to_host_bytes_flattens_multidimensional_input():
    z = np.zeros((2, 3, 4), dtype=np.uint8)

    result = to_host_bytes(z)

    assert result.shape == (24,)


def test_memcpy_host_to_device_computes_correct_nbytes_and_direction(monkeypatch):

    import utils.cuda_handler as trt_utils
    import tests.unit.utils.cuda.cudart as cudart

    calls = []
    monkeypatch.setattr(
        cudart,
        "cudaMemcpy",
        lambda *args: (calls.append(args), (cudart.cudaError_t.cudaSuccess,))[1],
    )

    host_arr = np.zeros(4, dtype=np.float32)
    trt_utils.memcpy_host_to_device(device_ptr=555, host_arr=host_arr)

    (device_ptr, arr, nbytes, kind) = calls[0]
    assert device_ptr == 555
    assert nbytes == 16
    assert kind == cudart.cudaMemcpyKind.cudaMemcpyHostToDevice


def test_memcpy_device_to_host_computes_correct_nbytes_and_direction(monkeypatch):

    import utils.cuda_handler as trt_utils
    import tests.unit.utils.cuda.cudart as cudart

    calls = []
    monkeypatch.setattr(
        cudart,
        "cudaMemcpy",
        lambda *args: (calls.append(args), (cudart.cudaError_t.cudaSuccess,))[1],
    )

    host_arr = np.zeros(4, dtype=np.float32)
    trt_utils.memcpy_device_to_host(host_arr=host_arr, device_ptr=777)

    (dst, src, nbytes, kind) = calls[0]
    assert src == 777
    assert nbytes == 16
    assert kind == cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
