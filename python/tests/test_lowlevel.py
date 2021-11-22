"""
Tests for the low-level C interface
"""
import os
import platform
import tempfile
import unittest

import numpy as np

import _kastore

IS_WINDOWS = platform.system() == "Windows"


@unittest.skipIf(IS_WINDOWS, "Not worth making this work on windows")
class TestBasicOperation(unittest.TestCase):
    """
    Simple tests just to verify the basic operation.
    """

    def assertDataEqual(self, d1, d2):
        self.assertEqual(d1.keys(), d2.keys())
        for key in d1.keys():
            np.testing.assert_array_equal(d1[key], d2[key])

    def test_file_round_trip(self):
        with tempfile.TemporaryFile() as f:
            data = {"a": np.arange(10)}
            _kastore.dump(data, f)
            f.seek(0)
            x = _kastore.load(f)
            self.assertDataEqual(x, data)

    def test_fileno_round_trip(self):
        with tempfile.TemporaryFile() as f:
            data = {"a": np.arange(10)}
            _kastore.dump(data, f.fileno())
            f.seek(0)
            x = _kastore.load(f.fileno())
            self.assertDataEqual(x, data)

    def test_same_file_round_trip(self):
        with tempfile.TemporaryFile() as f:
            for j in range(10):
                start = f.tell()
                data = {"a": np.arange(j * 100)}
                _kastore.dump(data, f)
                f.seek(start)
                x = _kastore.load(f)
                self.assertDataEqual(x, data)


@unittest.skipIf(IS_WINDOWS, "Not worth making this work on windows")
class TestInputs(unittest.TestCase):
    def test_bad_numeric_fd(self):
        for fd in ["1", 1.0, 2.0]:
            with self.assertRaises(TypeError):
                _kastore.dump({}, fd)
            with self.assertRaises(TypeError):
                _kastore.load(fd)

    def test_bad_fd(self):
        bad_fd = 10000
        with self.assertRaises(OSError):
            _kastore.dump({}, bad_fd)
        with self.assertRaises(OSError):
            _kastore.load(bad_fd)
        bad_fd = -1
        with self.assertRaises(ValueError):
            _kastore.dump({}, bad_fd)
        with self.assertRaises(ValueError):
            _kastore.load(bad_fd)

    def test_bad_file_mode(self):
        with open(os.devnull) as f:
            with self.assertRaises(OSError):
                _kastore.dump({}, f)
        with open(os.devnull, "w") as f:
            with self.assertRaises(OSError):
                _kastore.load(f)

    def test_bad_load_args(self):
        with self.assertRaises(TypeError):
            _kastore.load(0, read_all="sdf")

    def test_bad_dtype(self):
        with open(os.devnull, "w") as f:
            with self.assertRaises(ValueError):
                # complex number
                array = np.array([0], dtype="c16")
                _kastore.dump({"a": array}, f)
