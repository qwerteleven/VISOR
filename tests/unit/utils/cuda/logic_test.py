import ml_dtypes
import numpy as np
import pytest
import tests.unit.utils.cuda.tensorrt as trt

from utils.cuda_handler import (
    GiB,
    add_help,
    find_sample_data,
    locate_files,
    _trt_dtype_to_np,
)


def test_gib_converts_correctly():
    assert GiB(1) == 1024**3
    assert GiB(2) == 2 * 1024**3
    assert GiB(0) == 0


def test_add_help_does_not_raise_with_no_extra_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    add_help("a description")  # should not raise


def test_add_help_ignores_unknown_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--some-unrelated-flag", "value"])
    add_help("a description")  # should not raise


def test_add_help_exits_on_help_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        add_help("a description")
    assert exc_info.value.code == 0


def test_locate_files_finds_file_in_single_directory(tmp_path):
    (tmp_path / "model.onnx").write_text("data")

    result = locate_files([str(tmp_path)], ["model.onnx"])

    assert result == [str(tmp_path / "model.onnx")]


def test_locate_files_uses_first_matching_directory(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "shared.txt").write_text("from a")
    (dir_b / "shared.txt").write_text("from b")

    result = locate_files([str(dir_a), str(dir_b)], ["shared.txt"])

    assert result == [str(dir_a / "shared.txt")]


def test_locate_files_raises_when_file_not_found_anywhere(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        locate_files([str(tmp_path)], ["missing.txt"])


def test_locate_files_finds_multiple_files_across_directories(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "one.txt").write_text("1")
    (dir_b / "two.txt").write_text("2")

    result = locate_files([str(dir_a), str(dir_b)], ["one.txt", "two.txt"])

    assert result == [str(dir_a / "one.txt"), str(dir_b / "two.txt")]


def test_locate_files_error_message_includes_custom_err_msg(tmp_path):
    with pytest.raises(FileNotFoundError, match="custom hint here"):
        locate_files([str(tmp_path)], ["missing.txt"], err_msg="custom hint here")


def test_find_sample_data_finds_files_in_custom_datadir(tmp_path, monkeypatch):
    (tmp_path / "weights.bin").write_text("data")
    monkeypatch.setattr("sys.argv", ["prog", "-d", str(tmp_path)])

    data_paths, found = find_sample_data(find_files=["weights.bin"])

    assert found == [str(tmp_path / "weights.bin")]


def test_find_sample_data_falls_back_to_datadir_when_subfolder_missing(
    tmp_path, monkeypatch
):
    (tmp_path / "weights.bin").write_text("data")
    monkeypatch.setattr("sys.argv", ["prog", "-d", str(tmp_path)])

    data_paths, found = find_sample_data(
        subfolder="does_not_exist", find_files=["weights.bin"]
    )

    assert found == [str(tmp_path / "weights.bin")]


def test_trt_dtype_to_np_maps_native_types_directly():
    assert _trt_dtype_to_np(trt.DataType.FLOAT) == np.dtype("float32")
    assert _trt_dtype_to_np(trt.DataType.INT32) == np.dtype("int32")
    assert _trt_dtype_to_np(trt.DataType.BOOL) == np.dtype("bool")


def test_trt_dtype_to_np_falls_back_to_ml_dtypes_for_bf16():
    result = _trt_dtype_to_np(trt.DataType.BF16)
    assert result == np.dtype(ml_dtypes.bfloat16)


def test_trt_dtype_to_np_falls_back_to_ml_dtypes_for_fp8_e4m3():
    result = _trt_dtype_to_np(trt.DataType.E4M3)
    assert result == np.dtype(ml_dtypes.float8_e4m3)


def test_trt_dtype_to_np_falls_back_to_ml_dtypes_for_fp8_e5m2():
    result = _trt_dtype_to_np(trt.DataType.E5M2)
    assert result == np.dtype(ml_dtypes.float8_e5m2)


def test_trt_dtype_to_np_raises_for_truly_unmapped_type():
    import enum

    class FakeUnmappedDtype(enum.Enum):
        MYSTERY = "mystery"

    with pytest.raises(TypeError, match="No numpy/ml_dtypes mapping"):
        _trt_dtype_to_np(FakeUnmappedDtype.MYSTERY)
