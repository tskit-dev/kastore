"""
Basic tests for the information API.
"""
import io
import os
import pathlib
import tempfile
import unittest

import numpy as np

import kastore as kas
import kastore.exceptions as exceptions


class InterfaceTest(unittest.TestCase):
    """
    Superclass of tests that assess the kastore module interface.
    """

    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_info")
        os.close(fd)
        self.temp_file = path

    def tearDown(self):
        try:
            os.unlink(self.temp_file)
        except OSError:
            pass

    def verify_dicts_equal(self, d1, d2):
        self.assertEqual(sorted(d1.keys()), sorted(d2.keys()))
        for key in d1.keys():
            np.testing.assert_equal(d1[key], d2[key])


class TestBasicInfo(InterfaceTest):
    """
    Check that the info we return is accurate.
    """

    def verify(self, data):
        kas.dump(data, self.temp_file)
        for read_all in [True, False]:
            new_data = kas.load(self.temp_file, read_all=read_all)
            for key, array in new_data.items():
                info = new_data.info(key)
                s = str(info)
                self.assertGreater(len(s), 0)
                self.assertEqual(array.nbytes, info.size)
                self.assertEqual(array.shape, info.shape)
                self.assertEqual(array.dtype, np.dtype(info.dtype))

    def test_all_dtypes(self):
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
            self.verify(data)


class TestClosedStore(InterfaceTest):
    """
    Checks that a closed store is no longer accessible.
    """

    def verify_closed(self, store):
        self.assertRaises(exceptions.StoreClosedError, store.get, "a")
        self.assertRaises(exceptions.StoreClosedError, store.info, "a")
        self.assertRaises(exceptions.StoreClosedError, list, store.keys())
        self.assertRaises(exceptions.StoreClosedError, list, store.items())

    def test_context_manager(self):
        N = 100
        data = {"a": np.arange(N)}
        kas.dump(data, self.temp_file)
        with kas.load(self.temp_file) as store:
            self.assertIn("a", store)
            self.assertTrue(np.array_equal(store["a"], np.arange(N)))
        self.verify_closed(store)

    def test_manual_close(self):
        N = 100
        data = {"a": np.arange(N)}
        kas.dump(data, self.temp_file)
        store = kas.load(self.temp_file)
        self.assertIn("a", store)
        self.assertTrue(np.array_equal(store["a"], np.arange(N)))
        store.close()
        self.verify_closed(store)


class StoreRoundTripMixin:
    """
    We should be able to feed the output of load directly to dump.
    """

    def verify(self, data):
        kas.dump(data, self.temp_file, engine=self.engine)
        copy = kas.load(self.temp_file, engine=self.engine)
        kas.dump(copy, self.temp_file, engine=self.engine)
        copy = kas.load(self.temp_file, engine=self.engine)
        self.verify_dicts_equal(copy, data)

    def test_empty(self):
        self.verify({})

    def test_non_empty(self):
        self.verify({"a": np.arange(10), "b": np.zeros(100)})


class TestStoreRoundTripPyEngine(StoreRoundTripMixin, InterfaceTest):
    engine = kas.PY_ENGINE


class TestStoreRoundTripCEngine(StoreRoundTripMixin, InterfaceTest):
    engine = kas.C_ENGINE


class TestBytesIOInput(InterfaceTest):
    """
    Tests that we get the expected behaviour when using BytesIO.
    """

    def test_py_engine_single(self):
        data = {"a": np.arange(10), "b": np.zeros(100)}
        fileobj = io.BytesIO()
        kas.dump(data, fileobj, engine=kas.PY_ENGINE)
        fileobj.seek(0)
        data_2 = kas.load(fileobj, engine=kas.PY_ENGINE)
        self.verify_dicts_equal(data, data_2)

    def test_py_engine_multi(self):
        data = {"a": np.arange(10), "b": np.zeros(100)}
        n = 10
        fileobj = io.BytesIO()
        for _ in range(n):
            kas.dump(data, fileobj, engine=kas.PY_ENGINE)
        fileobj.seek(0)
        for _ in range(n):
            data_2 = kas.load(fileobj, read_all=True, engine=kas.PY_ENGINE)
            self.verify_dicts_equal(data, data_2)

    def test_c_engine_fails(self):
        data = {"a": np.arange(10), "b": np.zeros(100)}
        fileobj = io.BytesIO()
        with self.assertRaises(io.UnsupportedOperation):
            kas.dump(data, fileobj, engine=kas.C_ENGINE)
        with self.assertRaises(io.UnsupportedOperation):
            kas.load(fileobj, engine=kas.C_ENGINE)


class TestGetInclude(unittest.TestCase):
    """
    Check that the get_include works as expected.
    """

    def test_output(self):
        include_dir = kas.get_include()
        self.assertTrue(os.path.exists(include_dir))
        self.assertTrue(os.path.isdir(include_dir))
        path = os.path.join(kas.__path__[0], "include")
        self.assertEqual(include_dir, os.path.abspath(path))


class TestMissingCEngine(InterfaceTest):
    """
    Tests that we handle the missing low level module smoothly.
    """

    def test_dump(self):
        data = {"a": np.zeros(1)}
        try:
            kas._kastore_loaded = False
            with self.assertRaises(RuntimeError):
                kas.dump(data, self.temp_file, engine=kas.C_ENGINE)
        finally:
            kas._kastore_loaded = True

    def test_load(self):
        data = {"a": np.zeros(1)}
        kas.dump(data, self.temp_file)
        try:
            kas._kastore_loaded = False
            with self.assertRaises(RuntimeError):
                kas.load(self.temp_file, engine=kas.C_ENGINE)
        finally:
            kas._kastore_loaded = True


class TestOpenSemantics(unittest.TestCase):
    """
    Tests that we can open file-like objects with the correct semantics.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def verify_path(self, path):
        with kas._open_file(path, "w") as f:
            f.write("xyz")
        with open(path) as f:
            self.assertEqual(f.read(), "xyz")

    def test_string(self):
        path = os.path.join(self.temp_dir.name, "testfile")
        self.verify_path(path)

    def test_bytes(self):
        path = os.path.join(self.temp_dir.name, "testfile")
        self.verify_path(path.encode())

    def test_pathlib(self):
        path = pathlib.Path(self.temp_dir.name) / "testfile"
        self.verify_path(path)

    def test_fd(self):
        path = os.path.join(self.temp_dir.name, "testfile")
        with open(path, "wb") as f:
            fd = f.fileno()
            stat_before = os.fstat(fd)
            with kas._open_file(fd, "wb") as f:
                f.write(b"xyz")
            stat_after = os.fstat(fd)
            self.assertEqual(stat_before.st_mode, stat_after.st_mode)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"xyz")

    def test_regular_file(self):
        path = os.path.join(self.temp_dir.name, "testfile")
        with open(path, "w") as f_input:
            with kas._open_file(f_input, "w") as f:
                f.write("xyz")
        with open(path) as f:
            self.assertEqual(f.read(), "xyz")

    def test_temp_file(self):
        with tempfile.TemporaryFile() as f_input:
            with kas._open_file(f_input, "wb") as f:
                f.write(b"xyz")
            f_input.seek(0)
            self.assertEqual(f_input.read(), b"xyz")

    def test_string_io(self):
        buff = io.StringIO()
        with kas._open_file(buff, "w") as f:
            f.write("xyz")
        self.assertEqual(buff.getvalue(), "xyz")

    def test_bytes_io(self):
        buff = io.BytesIO()
        with kas._open_file(buff, "wb") as f:
            f.write(b"xyz")
        self.assertEqual(buff.getvalue(), b"xyz")

    def test_non_file(self):
        for bad_type in [None, {}, []]:
            with self.assertRaises(TypeError):
                with kas._open_file(bad_type, "wb"):
                    pass


class TestExceptions(unittest.TestCase):
    def test_inheritance(self):
        # Make sure all our exceptions are subclasses of KastoreException.
        exceptions = [
            kas.FileFormatError,
            kas.VersionTooNewError,
            kas.VersionTooOldError,
            kas.StoreClosedError,
        ]
        for exc_class in exceptions:
            self.assertTrue(issubclass(exc_class, kas.KastoreException))

    def test_format_error(self):
        with self.assertRaises(kas.FileFormatError):
            kas.loads(b"x" * 1024)
        # This is also a KastoreException
        with self.assertRaises(kas.KastoreException):
            kas.loads(b"x" * 1024)
