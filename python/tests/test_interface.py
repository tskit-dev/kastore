import io
import os
import pathlib
import tempfile

import numpy as np
import pytest

import kastore as kas
import kastore.exceptions as exceptions

"""
Basic tests for the information API.
"""


@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_info")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def verify_dicts_equal(d1, d2):
    assert sorted(d1.keys()) == sorted(d2.keys())
    for key in d1.keys():
        np.testing.assert_equal(d1[key], d2[key])


def verify_basic_info(data, temp_file):
    kas.dump(data, temp_file)
    for read_all in [True, False]:
        new_data = kas.load(temp_file, read_all=read_all)
        for key, array in new_data.items():
            info = new_data.info(key)
            s = str(info)
            assert len(s) > 0
            assert array.nbytes == info.size
            assert array.shape == info.shape
            assert array.dtype == np.dtype(info.dtype)


def test_all_dtypes(temp_file):
    dtypes = [
        "int8",
        "uint8",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
    ]
    for n in range(10):
        data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
        verify_basic_info(data, temp_file)


def verify_closed(store):
    with pytest.raises(exceptions.StoreClosedError):
        store.get("a")
    with pytest.raises(exceptions.StoreClosedError):
        store.info("a")
    with pytest.raises(exceptions.StoreClosedError):
        list(store.keys())
    with pytest.raises(exceptions.StoreClosedError):
        list(store.items())


def test_context_manager(temp_file):
    N = 100
    data = {"a": np.arange(N)}
    kas.dump(data, temp_file)
    with kas.load(temp_file) as store:
        assert "a" in store
        assert np.array_equal(store["a"], np.arange(N))
    verify_closed(store)


def test_manual_close(temp_file):
    N = 100
    data = {"a": np.arange(N)}
    kas.dump(data, temp_file)
    store = kas.load(temp_file)
    assert "a" in store
    assert np.array_equal(store["a"], np.arange(N))
    store.close()
    verify_closed(store)


def verify_round_trip(data, temp_file, engine):
    kas.dump(data, temp_file, engine=engine)
    copy = kas.load(temp_file, engine=engine)
    kas.dump(copy, temp_file, engine=engine)
    copy = kas.load(temp_file, engine=engine)
    verify_dicts_equal(copy, data)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_round_trip_empty(temp_file, engine):
    verify_round_trip({}, temp_file, engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_round_trip_non_empty(temp_file, engine):
    verify_round_trip({"a": np.arange(10), "b": np.zeros(100)}, temp_file, engine)


def test_bytes_io_py_engine_single():
    data = {"a": np.arange(10), "b": np.zeros(100)}
    fileobj = io.BytesIO()
    kas.dump(data, fileobj, engine=kas.PY_ENGINE)
    fileobj.seek(0)
    data_2 = kas.load(fileobj, engine=kas.PY_ENGINE)
    verify_dicts_equal(data, data_2)


def test_bytes_io_py_engine_multi():
    data = {"a": np.arange(10), "b": np.zeros(100)}
    n = 10
    fileobj = io.BytesIO()
    for _ in range(n):
        kas.dump(data, fileobj, engine=kas.PY_ENGINE)
    fileobj.seek(0)
    for _ in range(n):
        data_2 = kas.load(fileobj, read_all=True, engine=kas.PY_ENGINE)
        verify_dicts_equal(data, data_2)


def test_bytes_io_c_engine_fails():
    data = {"a": np.arange(10), "b": np.zeros(100)}
    fileobj = io.BytesIO()
    with pytest.raises(io.UnsupportedOperation):
        kas.dump(data, fileobj, engine=kas.C_ENGINE)
    with pytest.raises(io.UnsupportedOperation):
        kas.load(fileobj, engine=kas.C_ENGINE)


def test_get_include_output():
    include_dir = kas.get_include()
    assert os.path.exists(include_dir)
    assert os.path.isdir(include_dir)
    path = os.path.join(kas.__path__[0], "include")
    assert include_dir == os.path.abspath(path)


def test_missing_c_engine_dump(temp_file):
    data = {"a": np.zeros(1)}
    try:
        kas._kastore_loaded = False
        with pytest.raises(RuntimeError):
            kas.dump(data, temp_file, engine=kas.C_ENGINE)
    finally:
        kas._kastore_loaded = True


def test_missing_c_engine_load(temp_file):
    data = {"a": np.zeros(1)}
    kas.dump(data, temp_file)
    try:
        kas._kastore_loaded = False
        with pytest.raises(RuntimeError):
            kas.load(temp_file, engine=kas.C_ENGINE)
    finally:
        kas._kastore_loaded = True


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    temp_dir.cleanup()


def verify_path(path):
    with kas._open_file(path, "w") as f:
        f.write("xyz")
    with open(path) as f:
        assert f.read() == "xyz"


def test_open_string(temp_dir):
    path = os.path.join(temp_dir.name, "testfile")
    verify_path(path)


def test_open_bytes(temp_dir):
    path = os.path.join(temp_dir.name, "testfile")
    verify_path(path.encode())


def test_open_pathlib(temp_dir):
    path = pathlib.Path(temp_dir.name) / "testfile"
    verify_path(path)


def test_open_fd(temp_dir):
    path = os.path.join(temp_dir.name, "testfile")
    with open(path, "wb") as f:
        fd = f.fileno()
        stat_before = os.fstat(fd)
        with kas._open_file(fd, "wb") as f:
            f.write(b"xyz")
        stat_after = os.fstat(fd)
        assert stat_before.st_mode == stat_after.st_mode
    with open(path, "rb") as f:
        assert f.read() == b"xyz"


def test_open_regular_file(temp_dir):
    path = os.path.join(temp_dir.name, "testfile")
    with open(path, "w") as f_input:
        with kas._open_file(f_input, "w") as f:
            f.write("xyz")
    with open(path) as f:
        assert f.read() == "xyz"


def test_open_temp_file():
    with tempfile.TemporaryFile() as f_input:
        with kas._open_file(f_input, "wb") as f:
            f.write(b"xyz")
        f_input.seek(0)
        assert f_input.read() == b"xyz"


def test_open_string_io():
    buff = io.StringIO()
    with kas._open_file(buff, "w") as f:
        f.write("xyz")
    assert buff.getvalue() == "xyz"


def test_open_bytes_io():
    buff = io.BytesIO()
    with kas._open_file(buff, "wb") as f:
        f.write(b"xyz")
    assert buff.getvalue() == b"xyz"


@pytest.mark.parametrize("bad_type", [None, {}, []])
def test_open_non_file(bad_type):
    with pytest.raises(TypeError):
        with kas._open_file(bad_type, "wb"):
            pass


@pytest.mark.parametrize(
    "exc_class",
    [
        kas.FileFormatError,
        kas.VersionTooNewError,
        kas.VersionTooOldError,
        kas.StoreClosedError,
    ],
)
def test_exception_inheritance(exc_class):
    assert issubclass(exc_class, kas.KastoreException)


def test_format_error():
    with pytest.raises(kas.FileFormatError):
        kas.loads(b"x" * 1024)
    # This is also a KastoreException
    with pytest.raises(kas.KastoreException):
        kas.loads(b"x" * 1024)
