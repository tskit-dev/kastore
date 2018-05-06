"""
Basic tests for the information API.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import tempfile
import os

import six
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
        for use_mmap in [True, False]:
            new_data = kas.load(self.temp_file, use_mmap=use_mmap)
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
        self.assertRaises(exceptions.StoreClosedError, list, six.iterkeys(store))
        self.assertRaises(exceptions.StoreClosedError, list, six.iteritems(store))

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

    def test_arrays_mmap(self):
        # Make sure that arrays returned are numpy mmap objects,
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.memmap.html
        N = 100
        data = {"a": np.arange(N)}
        kas.dump(data, self.temp_file)
        with kas.load(self.temp_file) as store:
            new_array = store["a"]
            self.assertTrue(np.array_equal(new_array, data["a"]))
            self.assertEqual(new_array.filename, self.temp_file)
            self.assertEqual(new_array.mode, 'r')

    def test_arrays_read_only(self):
        N = 10
        data = {"a": np.arange(N, dtype=int)}
        kas.dump(data, self.temp_file)
        with kas.load(self.temp_file) as store:
            new_array = store["a"]
            self.assertTrue(np.array_equal(new_array, data["a"]))
            self.assertFalse(new_array.flags.writeable)

    def test_arrays_after_close(self):
        N = 100
        data = {"a": np.arange(N)}
        kas.dump(data, self.temp_file)
        for use_mmap in [True, False]:
            with kas.load(self.temp_file, use_mmap=False) as store:
                new_array = store["a"]
                self.assertTrue(np.array_equal(new_array, data["a"]))
            self.assertTrue(np.array_equal(new_array, data["a"]))
