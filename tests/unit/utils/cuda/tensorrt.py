import enum
from math import prod


class NetworkDefinitionCreationFlag(enum.IntEnum):
    EXPLICIT_BATCH = 0


class DataType(enum.Enum):
    FLOAT = "float32"
    HALF = "float16"
    INT8 = "int8"
    INT32 = "int32"
    BOOL = "bool"
    BF16 = "bf16"
    E4M3 = "e4m3"
    E5M2 = "e5m2"


class TensorIOMode(enum.IntEnum):
    INPUT = 0
    OUTPUT = 1


_NPTYPE_MAP = {
    DataType.FLOAT: "float32",
    DataType.HALF: "float16",
    DataType.INT8: "int8",
    DataType.INT32: "int32",
    DataType.BOOL: "bool",
}


def nptype(dtype):
    if dtype in _NPTYPE_MAP:
        return _NPTYPE_MAP[dtype]
    raise TypeError(f"no numpy dtype for {dtype}")


def volume(shape):
    return prod(shape) if len(shape) else 1


class ICudaEngine:
    pass
