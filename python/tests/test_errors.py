import os
import platform
import struct
import tempfile

import numpy as np
import pytest

import kastore as kas
import kastore.store as store

"""
Tests for error conditions.
"""

IS_WINDOWS = platform.system() == "Windows"


@pytest.fixture
def error_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_errors")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("bad_dict", [[], "w34", None, 1])
def test_bad_dicts(error_temp_file, engine, bad_dict):
    with pytest.raises(TypeError):
        kas.dump(bad_dict, error_temp_file, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("bad_filename", [[], None, {}])
def test_bad_filename_type(bad_filename, engine):
    with pytest.raises(TypeError):
        kas.dump({}, bad_filename, engine=engine)
    with pytest.raises(TypeError):
        kas.load(bad_filename, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("bad_key", [(1234,), b"1234", None, 1234])
def test_bad_keys(error_temp_file, engine, bad_key):
    a = np.zeros(1)
    with pytest.raises(TypeError):
        kas.dump({bad_key: a}, error_temp_file, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_bad_arrays(error_temp_file, engine):
    kas.dump({"a": []}, error_temp_file, engine=engine)
    for bad_array in [kas, lambda x: x, "1234", None, [[0, 1], [0, 2]]]:
        with pytest.raises(ValueError):
            kas.dump({"a": bad_array}, error_temp_file, engine=engine)
    # TODO add tests for arrays in fortran order and so on.


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("bad_file", ["no_such_file", "/no/such/file"])
def test_file_not_found(engine, bad_file):
    a = np.zeros(1)
    with pytest.raises(FileNotFoundError):
        kas.load(bad_file, engine=engine)
    with pytest.raises(FileNotFoundError):
        kas.dump({"a": a}, "/no/such/file", engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_file_is_a_directory(engine):
    tmp_dir = tempfile.mkdtemp()
    try:
        exception = IsADirectoryError
        if IS_WINDOWS:
            exception = PermissionError
        with pytest.raises(exception):
            kas.dump({"a": []}, tmp_dir, engine=engine)
        with pytest.raises(exception):
            kas.load(tmp_dir, engine=engine)
    finally:
        os.rmdir(tmp_dir)


@pytest.mark.parametrize("bad_engine", [None, {}, "no such engine", b"not an engine"])
def test_bad_engine_dump(bad_engine):
    with pytest.raises(ValueError):
        kas.dump({}, "", engine=bad_engine)


@pytest.mark.parametrize("bad_engine", [None, {}, "no such engine", b"not an engine"])
def test_bad_engine_load(bad_engine):
    with pytest.raises(ValueError):
        kas.load("", engine=bad_engine)


@pytest.fixture
def malformed_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_malformed_files")
    os.close(fd)
    yield path
    os.unlink(path)


def write_file(temp_file, num_items=0):
    data = {}
    for j in range(num_items):
        data["a" * (j + 1)] = np.arange(j)
    kas.dump(data, temp_file)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
def test_empty_file(malformed_temp_file, engine, read_all):
    with open(malformed_temp_file, "w"):
        pass
    assert os.path.getsize(malformed_temp_file) == 0
    with pytest.raises(EOFError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
def test_bad_magic(malformed_temp_file, engine, read_all):
    write_file(malformed_temp_file)
    with open(malformed_temp_file, "rb") as f:
        buff = bytearray(f.read())
    before_len = len(buff)
    buff[0:8] = b"12345678"
    assert len(buff) == before_len
    with open(malformed_temp_file, "wb") as f:
        f.write(buff)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("num_items", range(10))
@pytest.mark.parametrize("offset", [-2, -1, 1, 2**10])
def test_bad_file_size(malformed_temp_file, engine, read_all, num_items, offset):
    write_file(malformed_temp_file, num_items)
    file_size = os.path.getsize(malformed_temp_file)
    with open(malformed_temp_file, "rb") as f:
        buff = bytearray(f.read())
    before_len = len(buff)
    buff[16:24] = struct.pack("<Q", file_size + offset)
    assert len(buff) == before_len
    with open(malformed_temp_file, "wb") as f:
        f.write(buff)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("num_items", range(2, 5))
def test_truncated_file_descriptors(malformed_temp_file, engine, read_all, num_items):
    write_file(malformed_temp_file, num_items)
    with open(malformed_temp_file, "rb") as f:
        buff = bytearray(f.read())
    with open(malformed_temp_file, "wb") as f:
        f.write(buff[: num_items * store.ITEM_DESCRIPTOR_SIZE - 1])
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("num_items", range(2, 5))
def test_truncated_file_data(malformed_temp_file, engine, read_all, num_items):
    write_file(malformed_temp_file, num_items)
    with open(malformed_temp_file, "rb") as f:
        buff = bytearray(f.read())
    with open(malformed_temp_file, "wb") as f:
        f.write(buff[:-1])
    with pytest.raises(kas.FileFormatError):
        # Must call dict to ensure all the keys are loaded.
        dict(kas.load(malformed_temp_file, engine=engine, read_all=read_all))


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize(
    "bad_type",
    [len(store.np_dtype_to_type_map) + 1, 2 * len(store.np_dtype_to_type_map)],
)
def test_bad_item_types(malformed_temp_file, engine, read_all, bad_type):
    items = {"a": []}
    descriptors, file_size = store.pack_items(items)
    with open(malformed_temp_file, "wb") as f:
        descriptors[0].type = bad_type
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("offset", [-1, +1, 2, 100])
def test_bad_key_initial_offsets(malformed_temp_file, engine, read_all, offset):
    items = {"a": np.arange(100)}
    # First key offset must be at header_size + n * (descriptor_size)
    descriptors, file_size = store.pack_items(items)
    descriptors[0].key_start += offset
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("offset", [-1, +1, 2, 100])
def test_bad_key_non_sequential(malformed_temp_file, engine, read_all, offset):
    items = {"a": np.arange(100), "b": []}
    # Keys must be packed sequentially.
    descriptors, file_size = store.pack_items(items)
    descriptors[1].key_start += offset
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("offset", [-100, -1, +1, 2, 8, 16, 100])
def test_bad_array_initial_offset(malformed_temp_file, engine, read_all, offset):
    items = {"a": np.arange(100)}
    # First key offset must be at header_size + n * (descriptor_size)
    descriptors, file_size = store.pack_items(items)
    descriptors[0].array_start += offset
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("offset", [-1, 1, 2, -8, 8, 100])
def test_bad_array_non_sequential(malformed_temp_file, engine, read_all, offset):
    items = {"a": np.arange(100), "b": []}
    descriptors, file_size = store.pack_items(items)
    descriptors[1].array_start += offset
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
def test_bad_array_alignment(malformed_temp_file, engine, read_all):
    items = {"a": np.arange(100, dtype=np.int8), "b": []}
    descriptors, file_size = store.pack_items(items)
    descriptors[0].array_start += 1
    descriptors[0].array_len -= 1
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
def test_bad_array_packing(malformed_temp_file, engine, read_all):
    items = {"a": np.arange(100, dtype=np.int8), "b": []}
    descriptors, file_size = store.pack_items(items)
    descriptors[0].array_start += 8
    descriptors[0].array_len -= 8
    with open(malformed_temp_file, "wb") as f:
        store.write_file(f, descriptors, file_size)
    with pytest.raises(kas.FileFormatError):
        kas.load(malformed_temp_file, engine=engine, read_all=read_all)


def verify_major_version(temp_file, version, engine, read_all):
    write_file(temp_file)
    with open(temp_file, "rb") as f:
        buff = bytearray(f.read())
    before_len = len(buff)
    buff[8:10] = struct.pack("<H", version)
    assert len(buff) == before_len
    with open(temp_file, "wb") as f:
        f.write(buff)
    kas.load(temp_file, engine=engine, read_all=read_all)


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
def test_major_version_too_old(malformed_temp_file, engine, read_all):
    with pytest.raises(kas.VersionTooOldError):
        verify_major_version(
            malformed_temp_file, store.VERSION_MAJOR - 1, engine, read_all
        )


@pytest.mark.parametrize(
    "engine,read_all",
    [
        (kas.PY_ENGINE, False),
        (kas.C_ENGINE, False),
        (kas.PY_ENGINE, True),
        (kas.C_ENGINE, True),
    ],
)
@pytest.mark.parametrize("j", range(1, 5))
def test_major_version_too_new(malformed_temp_file, engine, read_all, j):
    with pytest.raises(kas.VersionTooNewError):
        verify_major_version(
            malformed_temp_file, store.VERSION_MAJOR + j, engine, read_all
        )
