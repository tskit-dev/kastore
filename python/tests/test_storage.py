import os
import tempfile

import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hst
import numpy as np
import pytest

import kastore as kas


"""
Basic tests for the storage integrity of the data.
"""


# Set the deadline to None to avoid weird behaviour on CI.
hypothesis.settings.register_profile(
    "kastore_defaults",
    deadline=None,
    # Supress warnings resultsing from inheritance
    suppress_health_check=(hypothesis.HealthCheck.differing_executors,),
)
hypothesis.settings.load_profile("kastore_defaults")

# Exclude any 'other' unicode categories:
# http://www.unicode.org/reports/tr44/#General_Category_Values
key_alphabet = hst.characters(blacklist_categories=("C",))


@pytest.fixture
def storage_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_input")
    os.close(fd)
    yield path
    os.unlink(path)


def test_empty_key(storage_temp_file):
    with pytest.raises(ValueError):
        kas.dump({"": np.zeros(1)}, storage_temp_file)


def test_2d_array(storage_temp_file):
    with pytest.raises(ValueError):
        kas.dump({"a": np.zeros((1, 1))}, storage_temp_file)


def test_3d_array(storage_temp_file):
    with pytest.raises(ValueError):
        kas.dump({"a": np.zeros((2, 2, 1))}, storage_temp_file)


@pytest.fixture
def roundtrip_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    yield path
    os.unlink(path)


def verify_roundtrip(data, temp_file):
    for engine in [kas.C_ENGINE, kas.PY_ENGINE]:
        for read_all in [True, False]:
            kas.dump(data, temp_file, engine=engine)
            new_data = kas.load(temp_file, read_all=read_all, engine=engine)
            assert sorted(new_data.keys()) == sorted(data.keys())
            for key, source_array in data.items():
                dest_array = new_data[key]
                # Numpy's testing assert_equal will deal correctly with NaNs.
                np.testing.assert_equal(source_array, dest_array)
            # Make sure the file is closed before opening it again.
            del new_data


def verify_bytes_roundtrip(data):
    encoded = kas.dumps(data)
    new_data = kas.loads(encoded)
    assert sorted(new_data.keys()) == sorted(data.keys())
    for key, source_array in data.items():
        dest_array = new_data[key]
        # Numpy's testing assert_equal will deal correctly with NaNs.
        np.testing.assert_equal(source_array, dest_array)


def test_roundtrip_zero_keys(roundtrip_temp_file):
    verify_roundtrip({}, roundtrip_temp_file)


def test_roundtrip_single_key(roundtrip_temp_file):
    verify_roundtrip({"a": np.zeros(1)}, roundtrip_temp_file)


def test_roundtrip_many_keys(roundtrip_temp_file):
    data = {}
    for j in range(1):
        data[str(j)] = j + np.zeros(j, dtype=np.uint32)
    verify_roundtrip(data, roundtrip_temp_file)


@pytest.mark.parametrize("n", range(10))
def test_roundtrip_all_dtypes(roundtrip_temp_file, n):
    dtypes = [
        "int8",
        "uint8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
    ]
    data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
    verify_roundtrip(data, roundtrip_temp_file)


def test_bytes_roundtrip_zero_keys():
    verify_bytes_roundtrip({})


def test_bytes_roundtrip_single_key():
    verify_bytes_roundtrip({"a": np.zeros(1)})


def test_bytes_roundtrip_many_keys():
    data = {}
    for j in range(1):
        data[str(j)] = j + np.zeros(j, dtype=np.uint32)
    verify_bytes_roundtrip(data)


@pytest.mark.parametrize("n", range(10))
def test_bytes_roundtrip_all_dtypes(n):
    dtypes = [
        "int8",
        "uint8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
    ]
    data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
    verify_bytes_roundtrip(data)


@hypothesis.given(key=hst.text(alphabet=key_alphabet, min_size=1))
def test_roundtrip_single_key_hypothesis(key):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({key: np.zeros(1)}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(
    keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1)
)
def test_roundtrip_many_keys_hypothesis(keys):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        data = {key: np.ones(j) * j for j, key in enumerate(keys)}
        verify_roundtrip(data, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(key=hst.text(alphabet=key_alphabet, min_size=1))
def test_bytes_roundtrip_single_key_hypothesis(key):
    verify_bytes_roundtrip({key: np.zeros(1)})


@hypothesis.given(
    keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1)
)
def test_bytes_roundtrip_many_keys_hypothesis(keys):
    data = {key: np.ones(j) * j for j, key in enumerate(keys)}
    verify_bytes_roundtrip(data)


shape_strategy = hnp.array_shapes(max_dims=1)


@hypothesis.given(value=hnp.arrays(dtype=np.uint8, shape=shape_strategy))
def test_roundtrip_single_uint8(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.int8, shape=shape_strategy))
def test_roundtrip_single_int8(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.int16, shape=shape_strategy))
def test_roundtrip_single_int16(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.uint16, shape=shape_strategy))
def test_roundtrip_single_uint16(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.int32, shape=shape_strategy))
def test_roundtrip_single_int32(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.uint32, shape=shape_strategy))
def test_roundtrip_single_uint32(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.int64, shape=shape_strategy))
def test_roundtrip_single_int64(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.uint64, shape=shape_strategy))
def test_roundtrip_single_uint64(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.float32, shape=shape_strategy))
def test_roundtrip_single_float32(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.float64, shape=shape_strategy))
def test_roundtrip_single_float64(value):
    fd, temp_file = tempfile.mkstemp(prefix="kas_test_rt")
    os.close(fd)
    try:
        verify_roundtrip({"a": value}, temp_file)
    finally:
        os.unlink(temp_file)


@hypothesis.given(value=hnp.arrays(dtype=np.uint8, shape=shape_strategy))
def test_bytes_roundtrip_single_uint8(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.int8, shape=shape_strategy))
def test_bytes_roundtrip_single_int8(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.int16, shape=shape_strategy))
def test_bytes_roundtrip_single_int16(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.uint16, shape=shape_strategy))
def test_bytes_roundtrip_single_uint16(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.int32, shape=shape_strategy))
def test_bytes_roundtrip_single_int32(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.uint32, shape=shape_strategy))
def test_bytes_roundtrip_single_uint32(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.int64, shape=shape_strategy))
def test_bytes_roundtrip_single_int64(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.uint64, shape=shape_strategy))
def test_bytes_roundtrip_single_uint64(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.float32, shape=shape_strategy))
def test_bytes_roundtrip_single_float32(value):
    verify_bytes_roundtrip({"a": value})


@hypothesis.given(value=hnp.arrays(dtype=np.float64, shape=shape_strategy))
def test_bytes_roundtrip_single_float64(value):
    verify_bytes_roundtrip({"a": value})
