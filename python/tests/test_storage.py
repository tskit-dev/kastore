"""
Basic tests for the storage integrity of the data.
"""
from __future__ import print_function
from __future__ import division

import unittest
import tempfile
import os

import numpy as np

import kastore as kas


class TestRoundTrip(unittest.TestCase):
    """
    Simple round-trip tests for some hand crafted cases.
    """
    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_rt_test")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def verify(self, data):
        kas.dump(data, self.temp_file)
        new_data = kas.load(self.temp_file)
        self.assertEqual(sorted(new_data.keys()), sorted(data.keys()))
        for key, source_array in data.items():
            dest_array = new_data[key]
            self.assertTrue(np.array_equal(source_array, dest_array))

    def test_single_key(self):
        self.verify({"a": np.zeros(1)})

    def test_many_keys(self):
        data = {}
        for j in range(100):
            data[str(j)] = j + np.zeros(j, dtype=np.uint32)
        self.verify(data)
