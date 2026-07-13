import json
import pytest
import logging
import os
from unittest.mock import Mock

import utils.io as utils_io

from utils.io import (
    load_config,
    get_config,
    oldest_file_in_tree,
    set_logger,
    save_onnx,
    check_onnx,
)


def test_load_config_raises_on_missing_file():
    with pytest.raises(Exception, match="config file not exists"):
        load_config("/no/such/config.json")


def test_load_config_returns_parsed_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"key": "value", "n": 42}))

    result = load_config(str(config_file))

    assert result == {"key": "value", "n": 42}


def test_load_config_raises_unbound_local_error_on_invalid_json(tmp_path):
    config_file = tmp_path / "bad_config.json"
    config_file.write_text("{not valid json")

    with pytest.raises(UnboundLocalError):
        load_config(str(config_file))


def test_load_config_uses_default_path_argument(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"default": True}))

    result = load_config()

    assert result == {"default": True}


def test_get_config_raises_on_missing_file():
    with pytest.raises(Exception, match="config file not exists"):
        get_config("/no/such/config.json", "logs")


def test_get_config_returns_requested_section(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"logs": {"MAX_LOGS": 5}, "other": {}}))

    result = get_config(str(config_file), "logs")

    assert result == {"MAX_LOGS": 5}


def test_get_config_raises_on_missing_section(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"other": {}}))

    with pytest.raises(Exception, match="section config not exists"):
        get_config(str(config_file), "logs")


def test_get_config_raises_unbound_local_error_on_invalid_json(tmp_path):
    config_file = tmp_path / "bad_config.json"
    config_file.write_text("{not valid json")

    with pytest.raises(UnboundLocalError):
        get_config(str(config_file), "logs")


def test_get_config_raises_unbound_local_error_on_race_condition_deletion(
    tmp_path, monkeypatch
):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"logs": {}}')

    import builtins

    real_open = builtins.open

    def flaky_open(path, *args, **kwargs):
        if str(path) == str(config_file):
            raise FileNotFoundError("vanished between isfile() and open()")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)

    with pytest.raises(UnboundLocalError):
        get_config(str(config_file), "logs")


def touch(path, mtime_offset_seconds=0):
    path.write_text("data")
    if mtime_offset_seconds:
        current = os.stat(path).st_mtime
        new_time = current - mtime_offset_seconds
        os.utime(path, (new_time, new_time))


def test_raises_on_missing_folder():
    with pytest.raises(Exception, match="folder not exists"):
        oldest_file_in_tree("/no/such/folder")


def test_raises_value_error_when_no_matching_files_exist(tmp_path):
    with pytest.raises(Exception, match="Can not get oldest file error"):
        oldest_file_in_tree(str(tmp_path), ".log")


def test_returns_the_single_matching_file(tmp_path):
    log_file = tmp_path / "only.log"
    touch(log_file)

    result = oldest_file_in_tree(str(tmp_path), ".log")

    assert result == str(log_file)


def test_returns_the_oldest_of_several_files(tmp_path):
    newer = tmp_path / "newer.log"
    older = tmp_path / "older.log"
    touch(newer, mtime_offset_seconds=0)
    touch(older, mtime_offset_seconds=3600)  # 1 hour older

    result = oldest_file_in_tree(str(tmp_path), ".log")

    assert result == str(older)


def test_ignores_files_with_a_different_extension(tmp_path):
    log_file = tmp_path / "keep.log"
    other_file = tmp_path / "ignore.txt"
    touch(other_file, mtime_offset_seconds=3600)
    touch(log_file)

    result = oldest_file_in_tree(str(tmp_path), ".log")

    assert result == str(log_file)


def test_searches_recursively_into_subfolders(tmp_path):
    subfolder = tmp_path / "nested"
    subfolder.mkdir()
    nested_log = subfolder / "deep.log"
    touch(nested_log, mtime_offset_seconds=3600)
    top_log = tmp_path / "top.log"
    touch(top_log)

    result = oldest_file_in_tree(str(tmp_path), ".log")

    assert result == str(nested_log)


def test_default_extension_is_dot_log(tmp_path):
    (tmp_path / "keep.log").write_text("data")
    (tmp_path / "ignore.txt").write_text("data")

    result = oldest_file_in_tree(str(tmp_path))  # no extension argument

    assert result.endswith(".log")


def test_raises_when_log_folder_missing(fake_logs_config, clean_root_logger):
    with pytest.raises(Exception, match="logs folder not exists"):
        set_logger("/no/such/log/folder", "myservice")


def test_creates_log_file_with_expected_name_pattern(
    tmp_path, fake_logs_config, clean_root_logger
):
    set_logger(str(tmp_path), "myservice")

    log_files = list(tmp_path.glob("myservice_*.log"))
    assert len(log_files) == 1


def test_adds_file_handler_to_root_logger(
    tmp_path, fake_logs_config, clean_root_logger
):
    before_count = len(clean_root_logger.handlers)

    set_logger(str(tmp_path), "myservice")

    assert len(clean_root_logger.handlers) == before_count + 1
    assert isinstance(clean_root_logger.handlers[-1], logging.FileHandler)
    assert clean_root_logger.level == logging.INFO


def test_removes_oldest_log_when_max_logs_exceeded(
    tmp_path, monkeypatch, clean_root_logger
):
    def fake_get_config(path, section):
        return {
            "time_format": "%Y%m%d",
            "log_format": "%(message)s",
            "MAX_LOGS": 2,
        }

    monkeypatch.setattr(utils_io, "get_config", fake_get_config)

    oldest = tmp_path / "old.log"
    oldest.write_text("data")
    middle = tmp_path / "middle.log"
    middle.write_text("data")
    newest = tmp_path / "newest.log"
    newest.write_text("data")

    # Make "old.log" genuinely the oldest by mtime
    old_time = os.stat(oldest).st_mtime - 3600
    os.utime(oldest, (old_time, old_time))

    set_logger(str(tmp_path), "myservice")

    assert not oldest.exists()  # oldest was removed
    assert middle.exists()
    assert newest.exists()


def test_raises_when_removing_oldest_log_fails(
    tmp_path, monkeypatch, clean_root_logger
):
    def fake_get_config(path, section):
        return {"time_format": "%Y%m%d", "log_format": "%(message)s", "MAX_LOGS": 0}

    monkeypatch.setattr(utils_io, "get_config", fake_get_config)

    (tmp_path / "existing.log").write_text("data")

    def broken_remove(path):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "remove", broken_remove)

    with pytest.raises(Exception, match="MAX_LOGS files reached"):
        set_logger(str(tmp_path), "myservice")


def test_save_onnx_removes_existing_output_file_before_saving(tmp_path, monkeypatch):
    output_file = tmp_path / "model.onnx"
    output_file.write_text("stale data")
    monkeypatch.setattr(utils_io.onnx, "save", Mock())

    save_onnx(str(output_file), object())
    assert not output_file.exists()


def test_save_onnx_removes_existing_external_data_file_before_saving(
    tmp_path, monkeypatch
):
    output_file = tmp_path / "model.onnx"
    data_file = tmp_path / "model.onnx.data"
    data_file.write_text("stale external tensor data")
    monkeypatch.setattr(utils_io.onnx, "save", Mock())

    save_onnx(str(output_file), object())

    assert not data_file.exists()


def test_save_onnx_does_not_error_when_no_existing_files(tmp_path, monkeypatch):
    output_file = tmp_path / "brand_new_model.onnx"
    monkeypatch.setattr(utils_io.onnx, "save", Mock())

    save_onnx(str(output_file), object())  # should not raise


def test_check_onnx_calls_checker_with_given_path(monkeypatch):
    fake_check = Mock()
    monkeypatch.setattr(utils_io.onnx.checker, "check_model", fake_check)

    check_onnx("some/model.onnx")

    fake_check.assert_called_once_with("some/model.onnx")


def test_check_onnx_reraises_on_invalid_model(monkeypatch):
    def broken_check(path):
        raise ValueError("invalid model structure")

    monkeypatch.setattr(utils_io.onnx.checker, "check_model", broken_check)

    with pytest.raises(ValueError, match="invalid model structure"):
        check_onnx("broken.onnx")


def test_check_onnx_prints_success_message_on_valid_model(monkeypatch, capsys):
    monkeypatch.setattr(utils_io.onnx.checker, "check_model", Mock())

    check_onnx("valid.onnx")

    out = capsys.readouterr().out
    assert "ONNX SUCCEEDED" in out


def test_check_onnx_prints_failure_message_before_reraising(monkeypatch, capsys):
    monkeypatch.setattr(
        utils_io.onnx.checker,
        "check_model",
        Mock(side_effect=ValueError("bad graph")),
    )

    with pytest.raises(ValueError):
        check_onnx("broken.onnx")

    out = capsys.readouterr().out
    assert "ONNX CHECK FAILED" in out
