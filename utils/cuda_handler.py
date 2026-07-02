#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import ctypes
from typing import Optional, List
import ml_dtypes
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


_RAW_CTYPE_BY_ITEMSIZE = {
    1: ctypes.c_uint8,
    2: ctypes.c_uint16,
    4: ctypes.c_uint32,
    8: ctypes.c_uint64,
}


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Runtime Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args, _ = parser.parse_known_args()


def find_sample_data(
    description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""
):
    """
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    """

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--datadir",
        help="Location of the TensorRT sample data directory, and any additional data directories.",
        action="append",
        default=[kDEFAULT_DATA_ROOT],
    )
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print(
                    f"WARNING: {data_path} does not exist. Trying {data_dir} instead."
                )
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print(
                "WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(
                    data_path
                )
            )
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)


def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError(
                "Could not find {:}. Searched in data paths: {:}\n{:}".format(
                    filename, data_paths, err_msg
                )
            )
    return found_files


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: np.dtype):
        dtype = np.dtype(dtype)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))

        try:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            self._host = np.ctypeslib.as_array(
                ctypes.cast(host_mem, pointer_type), (size,)
            )
        except NotImplementedError:
            # dtype has no ctypes equivalent (e.g. ml_dtypes.bfloat16) — build the view
            # using a same-itemsize raw integer type, then reinterpret (zero-copy) as
            # the real dtype so values are read/written with correct semantics.
            raw_ctype = _RAW_CTYPE_BY_ITEMSIZE[dtype.itemsize]
            pointer_type = ctypes.POINTER(raw_ctype)
            raw_array = np.ctypeslib.as_array(
                ctypes.cast(host_mem, pointer_type), (size,)
            )
            self._host = raw_array.view(dtype)

        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[: arr.size], arr.flat, casting="safe")

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def _trt_dtype_to_np(trt_dtype: trt.DataType) -> np.dtype:
    """
    Map a TensorRT DataType to a numpy-compatible dtype.
    Falls back to ml_dtypes for types numpy has no native representation for.
    """
    # types trt.nptype() already handles correctly
    try:
        return np.dtype(trt.nptype(trt_dtype))
    except TypeError:
        pass

    # explicit fallback table for types with no native numpy dtype
    fallback = {
        trt.DataType.BF16: ml_dtypes.bfloat16,
    }
    # newer TensorRT versions may also expose FP8 variants
    for attr_name in ("FP8", "E4M3", "E5M2"):
        fp8_dtype = getattr(trt.DataType, attr_name, None)
        if fp8_dtype is not None and trt_dtype == fp8_dtype:
            fallback[trt_dtype] = (
                ml_dtypes.float8_e4m3 if "E4M3" in attr_name else ml_dtypes.float8_e5m2
            )

    if trt_dtype in fallback:
        return np.dtype(fallback[trt_dtype])

    raise TypeError(
        f"No numpy/ml_dtypes mapping available for TensorRT dtype: {trt_dtype}"
    )


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(
    engine: trt.ICudaEngine,
    profile_idx: Optional[int] = None,
    stream: Optional[int] = None,
    context=None,
) -> tuple[list, list, list, int]:
    inputs, outputs, bindings = [], [], []

    if stream is None:
        stream = cuda_call(cudart.cudaStreamCreate())

    implicit_batch_mult = (
        engine.max_batch_size if engine.has_implicit_batch_dimension else 1
    )

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)

        shape = (
            engine.get_tensor_profile_shape(name, profile_idx)[-1]
            if profile_idx is not None
            else engine.get_tensor_shape(name)
        )

        if context is not None:
            shape = context.get_tensor_shape(name)

        if any(s < 0 for s in shape):
            raise ValueError(
                f"Tensor '{name}' has a dynamic dimension but no profile_idx was given. "
                f"Got shape: {tuple(shape)}"
            )

        size = trt.volume(shape) * implicit_batch_mult
        # dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        dtype = _trt_dtype_to_np(engine.get_tensor_dtype(name))
        mem = HostDeviceMem(size, dtype)  # uses cudaMallocHost internally

        bindings.append(int(mem.device))
        (
            inputs
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else outputs
        ).append(mem)

    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(
    inputs: List[HostDeviceMem],
    outputs: List[HostDeviceMem],
    stream: cudart.cudaStream_t,
):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


def _do_inference_base(inputs, outputs, stream, execute_async):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [
        cuda_call(
            cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)
        )
        for inp in inputs
    ]
    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [
        cuda_call(
            cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)
        )
        for out in outputs
    ]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)

    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


def get_input_output(engine):

    input_name_to_idx = {}
    output_name_to_idx = {}
    in_i, out_i = 0, 0
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name_to_idx[name] = in_i
            in_i += 1
        else:
            output_name_to_idx[name] = out_i
            out_i += 1

    return input_name_to_idx, output_name_to_idx


def to_host_bytes(z: np.array) -> np.array:
    """

        Convert any numpy/ml_dtypes array to a uint16/uint8 view
        safe for np.copyto into a pinned host buffer

    Args:
        z (np.array): cache matrix

    Returns:
        np.array: flat array cast  ml_dtypes.bfloat16 to np.uint16
    """

    if z.dtype == ml_dtypes.bfloat16:
        return z.view(np.uint16).ravel()

    return z.ravel()
