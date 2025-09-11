import os
import pathlib
import struct
import tempfile

import hypothesis
import hypothesis.strategies as hst
import numpy as np
import pytest

import kastore as kas
import kastore.store as store

"""
Tests checking that the file format is as it should be.
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


def test_file_signature_form():
    assert len(store.MAGIC) == 8
    assert store.MAGIC == b"\211KAS\r\n\032\n"


@pytest.fixture
def format_temp_file():
    fd, temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_rt_test")
    os.close(fd)
    yield temp_file
    os.unlink(temp_file)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("n", range(10))
def test_header_format(format_temp_file, engine, n):
    kas.dump(
        {str(j): np.zeros(1) for j in range(n)},
        format_temp_file,
        engine=engine,
    )
    with open(format_temp_file, "rb") as f:
        contents = f.read()
    assert contents[0:8] == store.MAGIC
    major, minor, num_items, size = struct.unpack("<HHIQ", contents[8:24])
    assert major == store.VERSION_MAJOR
    assert minor == store.VERSION_MINOR
    assert num_items == n
    assert size == len(contents)
    trailer = contents[24 : store.HEADER_SIZE]
    # The remainder should be zeros.
    assert trailer == bytearray(0 for _ in range(store.HEADER_SIZE - 24))


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_zero_items(format_temp_file, engine):
    kas.dump({}, format_temp_file, engine=engine)
    with open(format_temp_file, "rb") as f:
        contents = f.read()
    assert len(contents) == 64


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("n", range(10))
def test_item_descriptor_format(format_temp_file, engine, n):
    kas.dump(
        {str(j): j * np.ones(j) for j in range(n)},
        format_temp_file,
        engine=engine,
    )
    with open(format_temp_file, "rb") as f:
        contents = f.read()
    assert struct.unpack("<I", contents[12:16])[0] == n
    offset = store.HEADER_SIZE
    for _ in range(n):
        descriptor = contents[offset : offset + store.ITEM_DESCRIPTOR_SIZE]
        offset += store.ITEM_DESCRIPTOR_SIZE
        type_ = struct.unpack("<B", descriptor[0:1])[0]
        key_start, key_len, array_start, array_len = struct.unpack(
            "<QQQQ", descriptor[8:40]
        )
        trailer = descriptor[40 : store.ITEM_DESCRIPTOR_SIZE]
        # The remainder should be zeros.
        assert trailer == bytearray(0 for _ in range(store.ITEM_DESCRIPTOR_SIZE - 40))
        assert descriptor[1:4] == bytearray([0, 0, 0])
        assert type_ == store.FLOAT64
        assert key_start > 0
        assert key_len > 0
        assert array_start > 0
        assert array_len >= 0


def validate_storage(data, temp_file, engine):
    kas.dump(data, temp_file, engine=engine)
    with open(temp_file, "rb") as f:
        contents = f.read()
    offset = store.HEADER_SIZE
    descriptors = []
    for _ in range(len(data)):
        descriptor = store.ItemDescriptor.unpack(
            contents[offset : offset + store.ItemDescriptor.size]
        )
        descriptors.append(descriptor)
        offset += store.ItemDescriptor.size
    # Keys must be sorted lexicographically.
    sorted_keys = sorted(data.keys())
    # Keys should be packed sequentially immediately after the descriptors.
    offset = store.HEADER_SIZE + len(data) * store.ITEM_DESCRIPTOR_SIZE
    for d, key in zip(descriptors, sorted_keys):
        assert d.key_start == offset
        unpacked_key = contents[d.key_start : d.key_start + d.key_len]
        assert key.encode("utf8") == unpacked_key
        offset += d.key_len
    # Arrays should be packed sequentially immediately after the keys on
    # 8 byte boundaries
    for d, key in zip(descriptors, sorted_keys):
        remainder = offset % 8
        if remainder != 0:
            offset += 8 - remainder
        assert d.array_start == offset
        nbytes = d.array_len * store.type_size(d.type)
        array = np.frombuffer(
            contents[d.array_start : d.array_start + nbytes],
            dtype=store.type_to_np_dtype_map[d.type],
        )
        np.testing.assert_equal(data[key], array)
        offset += nbytes


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("n", range(10))
def test_simple_key_storage(format_temp_file, engine, n):
    validate_storage(
        {"a" * (j + 1): np.ones(1) for j in range(n)}, format_temp_file, engine
    )


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("n", range(10))
def test_simple_array_storage(format_temp_file, engine, n):
    validate_storage(
        {str(j): j * np.ones(j) for j in range(n)}, format_temp_file, engine
    )


# Note that hypothesis seems to be leaking memory, so when we're running tests
# against the C API for memory leaks this must be commented out.
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@hypothesis.given(
    keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1)
)
def test_many_keys(engine, keys):
    fd, temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_rt_test")
    os.close(fd)
    try:
        data = {key: np.ones(j, dtype=np.int32) * j for j, key in enumerate(keys)}
        validate_storage(data, temp_file, engine)
    finally:
        os.unlink(temp_file)


def test_descriptor_str():
    items = {"a": np.array([1], dtype=np.int8)}
    descriptors, file_size = store.pack_items(items)
    assert len(str(descriptors[0])) > 0


def test_single_key_1():
    items = {"a": np.array([1], dtype=np.int8)}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 128
    assert d.key_len == 1
    assert d.array_start == 136
    assert d.array_len == 1


def test_single_key_7():
    items = {"aaaaaaa": np.array([1], dtype=np.int8)}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 128
    assert d.key_len == 7
    assert d.array_start == 136
    assert d.array_len == 1


def test_single_key_8():
    items = {"aaaaaaaa": np.array([1], dtype=np.int8)}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 128
    assert d.key_len == 8
    assert d.array_start == 136
    assert d.array_len == 1


def test_two_keys_array_len1():
    a = np.array([1], dtype=np.int8)
    items = {"a": a, "b": a}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 192
    assert d.key_len == 1
    assert d.array_start == 200
    assert d.array_len == 1
    d = descriptors[1]
    assert d.key_start == 193
    assert d.key_len == 1
    assert d.array_start == 208
    assert d.array_len == 1


def test_two_keys_array_len8():
    a = np.array([1], dtype=np.int64)
    items = {"a": a, "b": a}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 192
    assert d.key_len == 1
    assert d.array_start == 200
    assert d.array_len == 1
    d = descriptors[1]
    assert d.key_start == 193
    assert d.key_len == 1
    assert d.array_start == 208
    assert d.array_len == 1


def test_two_keys_array_len4():
    a = np.array([1], dtype=np.int32)
    items = {"a": a, "b": a}
    descriptors, file_size = store.pack_items(items)
    d = descriptors[0]
    assert d.key_start == 192
    assert d.key_len == 1
    assert d.array_start == 200
    assert d.array_len == 1
    d = descriptors[1]
    assert d.key_start == 193
    assert d.key_len == 1
    assert d.array_start == 208
    assert d.array_len == 1


@pytest.fixture
def identity_temp_file():
    fd, temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_identity_test")
    os.close(fd)
    yield temp_file
    os.unlink(temp_file)


def verify_engine_identity(data, temp_file):
    kas.dump(data, temp_file, engine=kas.C_ENGINE)
    with open(temp_file, "rb") as f:
        c_file = f.read()
    kas.dump(data, temp_file, engine=kas.PY_ENGINE)
    with open(temp_file, "rb") as f:
        py_file = f.read()
    assert c_file == py_file


def test_engines_empty(identity_temp_file):
    verify_engine_identity({}, identity_temp_file)


def test_engines_one_key_empty_array(identity_temp_file):
    verify_engine_identity({"a": []}, identity_temp_file)


def test_engines_many_keys_empty_arrays(identity_temp_file):
    verify_engine_identity({"a" * (j + 1): [] for j in range(10)}, identity_temp_file)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint32, np.float64])
def test_engines_many_keys_nonempty_arrays(identity_temp_file, dtype):
    verify_engine_identity(
        {"a" * (j + 1): np.arange(j, dtype=dtype) for j in range(10)},
        identity_temp_file,
    )


def verify_all_dtypes(n, temp_file):
    dtypes = [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
    ]
    data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
    verify_engine_identity(data, temp_file)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_engines_all_dtypes(identity_temp_file, n):
    verify_all_dtypes(n, identity_temp_file)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_truncated_zero_bytes(engine):
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / "testfile"
        with open(path, "wb") as f:
            f.write(b"")
        assert path.exists()
        with pytest.raises(EOFError):
            kas.load(path, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_truncated_short_file(engine):
    example = {"a": np.arange(2)}
    encoded = kas.dumps(example)
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / "testfile"
        for j in range(1, 64):  # len(encoded) - 2):
            with open(path, "wb") as f:
                f.write(encoded[:j])
            assert path.exists()
            with pytest.raises(kas.FileFormatError):
                kas.load(path, engine=engine)
