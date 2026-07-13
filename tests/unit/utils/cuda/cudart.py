import ctypes
import enum


class cudaError_t(enum.IntEnum):
    cudaSuccess = 0
    cudaErrorInvalidValue = 1
    cudaErrorMemoryAllocation = 2


class cudaMemcpyKind(enum.IntEnum):
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2


class cudaStream_t(int):
    pass


_allocations = {}


def cudaMallocHost(nbytes):
    buf = ctypes.create_string_buffer(max(nbytes, 1))
    addr = ctypes.addressof(buf)
    _allocations[addr] = buf
    return (cudaError_t.cudaSuccess, addr)


def cudaMalloc(nbytes):
    buf = ctypes.create_string_buffer(max(nbytes, 1))
    addr = ctypes.addressof(buf)
    _allocations[addr] = buf
    return (cudaError_t.cudaSuccess, addr)


def cudaFree(ptr):
    _allocations.pop(ptr, None)
    return (cudaError_t.cudaSuccess,)


def cudaFreeHost(ptr):
    _allocations.pop(ptr, None)
    return (cudaError_t.cudaSuccess,)


def cudaMemcpy(dst, src, nbytes, kind):
    return (cudaError_t.cudaSuccess,)


def cudaMemcpyAsync(dst, src, nbytes, kind, stream):
    return (cudaError_t.cudaSuccess,)


def cudaStreamCreate():
    return (cudaError_t.cudaSuccess, cudaStream_t(1))


def cudaStreamDestroy(stream):
    return (cudaError_t.cudaSuccess,)


def cudaStreamSynchronize(stream):
    return (cudaError_t.cudaSuccess,)
