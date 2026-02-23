import os
import platform
import tempfile

import numpy as np
import pytest

import _kastore

"""
Tests for the low-level C interface
"""


IS_WINDOWS = platform.system() == "Windows"


def assert_data_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for key in d1.keys():
        np.testing.assert_array_equal(d1[key], d2[key])


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_file_round_trip():
    with tempfile.TemporaryFile() as f:
        data = {"a": np.arange(10)}
        _kastore.dump(data, f)
        f.seek(0)
        x = _kastore.load(f)
        assert_data_equal(x, data)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_fileno_round_trip():
    with tempfile.TemporaryFile() as f:
        data = {"a": np.arange(10)}
        _kastore.dump(data, f.fileno())
        f.seek(0)
        x = _kastore.load(f.fileno())
        assert_data_equal(x, data)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_same_file_round_trip():
    with tempfile.TemporaryFile() as f:
        for j in range(10):
            start = f.tell()
            data = {"a": np.arange(j * 100)}
            _kastore.dump(data, f)
            f.seek(start)
            x = _kastore.load(f)
            assert_data_equal(x, data)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
@pytest.mark.parametrize("fd", ["1", 1.0, 2.0])
def test_bad_numeric_fd(fd):
    with pytest.raises(TypeError):
        _kastore.dump({}, fd)
    with pytest.raises(TypeError):
        _kastore.load(fd)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_bad_fd():
    bad_fd = 10000
    with pytest.raises(OSError):
        _kastore.dump({}, bad_fd)
    with pytest.raises(OSError):
        _kastore.load(bad_fd)
    bad_fd = -1
    with pytest.raises(ValueError):
        _kastore.dump({}, bad_fd)
    with pytest.raises(ValueError):
        _kastore.load(bad_fd)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_bad_file_mode():
    with open(os.devnull) as f:
        with pytest.raises(OSError):
            _kastore.dump({}, f)
    with open(os.devnull, "w") as f:
        with pytest.raises(OSError):
            _kastore.load(f)


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_bad_load_args():
    with pytest.raises(TypeError):
        _kastore.load(0, read_all="sdf")


@pytest.mark.skipif(IS_WINDOWS, reason="Not worth making this work on windows")
def test_bad_dtype():
    with open(os.devnull, "w") as f:
        with pytest.raises(ValueError):
            # complex number
            array = np.array([0], dtype="c16")
            _kastore.dump({"a": array}, f)
