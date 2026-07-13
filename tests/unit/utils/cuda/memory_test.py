import ml_dtypes
import numpy as np
import pytest

from tests.unit.utils.cuda import cuda, cudart

from utils.cuda_handler import HostDeviceMem, check_cuda_err, cuda_call


def test_check_cuda_err_passes_on_cudart_success():
    check_cuda_err(cudart.cudaError_t.cudaSuccess)  # should not raise


def test_check_cuda_err_passes_on_valid_cuda_success():
    check_cuda_err(cuda.CUresult.CUDA_SUCCESS)  # should not raise


def test_check_cuda_err_raises_on_cuda_driver_error():
    with pytest.raises(RuntimeError, match="Cuda Error"):
        check_cuda_err(cuda.CUresult.CUDA_ERROR_INVALID_VALUE)


def test_check_cuda_err_raises_on_cuda_runtime_error():
    with pytest.raises(RuntimeError, match="Cuda Runtime Error"):
        check_cuda_err(cudart.cudaError_t.cudaErrorInvalidValue)


def test_check_cuda_err_raises_on_completely_unknown_type():
    with pytest.raises(RuntimeError, match="Unknown error type"):
        check_cuda_err("not a real error code")


def test_cuda_call_returns_single_result_unwrapped():
    result = cuda_call((cudart.cudaError_t.cudaSuccess, 42))
    assert result == 42


def test_cuda_call_returns_none_like_for_no_result_functions():
    result = cuda_call((cudart.cudaError_t.cudaSuccess,))
    assert result == ()


def test_cuda_call_raises_on_error_before_returning_result():
    with pytest.raises(RuntimeError):
        cuda_call((cudart.cudaError_t.cudaErrorMemoryAllocation, 0))


def test_host_device_mem_allocates_correct_size_and_dtype():
    mem = HostDeviceMem(size=10, dtype=np.float32)

    assert mem.host.shape == (10,)
    assert mem.host.dtype == np.float32
    assert mem.nbytes == 10 * 4
    assert isinstance(mem.device, int)

    mem.free()


def test_host_device_mem_host_memory_is_actually_writable():
    mem = HostDeviceMem(size=5, dtype=np.int32)

    mem.host[:] = [1, 2, 3, 4, 5]

    assert list(mem.host) == [1, 2, 3, 4, 5]
    mem.free()


def test_host_device_mem_setter_copies_values_in():
    mem = HostDeviceMem(size=4, dtype=np.float32)

    mem.host = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

    np.testing.assert_allclose(mem.host, [1.5, 2.5, 3.5, 4.5])
    mem.free()


def test_host_device_mem_setter_raises_when_array_too_large():
    mem = HostDeviceMem(size=2, dtype=np.float32)

    with pytest.raises(ValueError, match="Tried to fit"):
        mem.host = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # size 3 > 2

    mem.free()


def test_host_device_mem_bfloat16_fallback_view_roundtrips_correctly():
    mem = HostDeviceMem(size=4, dtype=ml_dtypes.bfloat16)

    assert mem.host.dtype == np.dtype(ml_dtypes.bfloat16)
    assert mem.host.shape == (4,)

    values = np.array([1.0, -2.5, 0.0, 100.0], dtype=ml_dtypes.bfloat16)
    mem.host[:] = values

    np.testing.assert_allclose(
        mem.host.astype(np.float32), values.astype(np.float32), rtol=1e-2
    )
    mem.free()


def test_host_device_mem_str_representation_does_not_raise():
    mem = HostDeviceMem(size=3, dtype=np.uint8)
    text = str(mem)
    assert "Host:" in text
    assert "Device:" in text
    mem.free()


def test_host_device_mem_repr_matches_str():
    mem = HostDeviceMem(size=3, dtype=np.uint8)
    assert repr(mem) == str(mem)
    mem.free()
