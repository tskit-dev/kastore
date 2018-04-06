"""
Tests for error conditions.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import tempfile
import os

import numpy as np

import _kastore
import kastore as kas


class TestLowLevelInterface(unittest.TestCase):
    """
    Exercise the low-level interface.
    """
    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_errors")
        os.close(fd)
        self.temp_file = path

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_bad_dicts(self):
        for bad_dict in [[], "w34", None, 1]:
            self.assertRaises(TypeError, _kastore.dump, bad_dict, "")
            self.assertRaises(TypeError, _kastore.dump, data=bad_dict, filename="")

    def test_bad_filename(self):
        for bad_filename in [[], None, {}, 1234]:
            self.assertRaises(TypeError, _kastore.dump, {}, bad_filename)
            self.assertRaises(TypeError, _kastore.dump, data={}, filename=bad_filename)
            self.assertRaises(TypeError, _kastore.load, bad_filename)
            self.assertRaises(TypeError, _kastore.load, filename=bad_filename)

    def test_bad_keys(self):
        a = np.zeros(1)
        for bad_key in [(1234,), b"1234", None, 1234]:
            self.assertRaises(
                TypeError, _kastore.dump, data={bad_key: a}, filename=self.temp_file)

    def test_bad_arrays(self):
        _kastore.dump(data={"a": []}, filename=self.temp_file)
        for bad_array in [_kastore, lambda x: x, "1234", None, [[0, 1], [0, 2]]]:
            self.assertRaises(
                ValueError, _kastore.dump, data={"a": bad_array},
                filename=self.temp_file)
        # TODO add tests for arrays in fortran order and so on.

    def test_bad_file(self):
        a = np.zeros(1)
        for bad_file in ["", "/no/such/file"]:
            # TODO Should raise the correct IO errors.
            self.assertRaises(
                ValueError, _kastore.dump, data={"a": a}, filename=bad_file)
            self.assertRaises(ValueError, _kastore.load, filename=bad_file)


class TestEngines(unittest.TestCase):
    """
    Check that we correctly identify bad engines
    """
    bad_engines = [None, {}, "no such engine", b"not an engine"]

    def test_bad_engine_dump(self):
        for bad_engine in self.bad_engines:
            self.assertRaises(ValueError, kas.dump, "", {}, engine=bad_engine)

    def test_bad_engine_load(self):
        for bad_engine in self.bad_engines:
            self.assertRaises(ValueError, kas.load, "", engine=bad_engine)
