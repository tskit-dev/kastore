"""
Basic tests for the information API.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import tempfile
import os

import numpy as np

import kastore as kas


class TestBasicInfo(unittest.TestCase):
    """
    Simple input errors.
    """
    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_info")
        os.close(fd)
        self.temp_file = path

    def verify(self, data):
        kas.dump(data, self.temp_file)
        for use_mmap in [True, False]:
            new_data = kas.load(self.temp_file, use_mmap=use_mmap)
            for key, array in new_data.items():
                info = new_data.info(key)
                s = str(info)
                self.assertGreater(len(s), 0)
                self.assertEqual(array.nbytes, info.size)
                self.assertEqual(array.dtype, np.dtype(info.dtype))

    def test_all_dtypes(self):
        dtypes = [
            "int8", "uint8", "uint32", "int32", "uint64", "int64", "float32", "float64"]
        for n in range(10):
            data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
            self.verify(data)
