"""
Basic tests for the information API.
"""
import unittest
import tempfile
import os

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
            "int8", "uint8", "uint32", "int32", "uint64", "int64", "float32", "float64"]
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
