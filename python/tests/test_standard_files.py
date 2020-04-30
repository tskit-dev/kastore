"""
Tests reading in the standard test files.
"""
import os.path
import unittest

import numpy as np

import kastore as kas


class StandardFilesMixin:
    """
    Read in the standard files.
    """

    @classmethod
    def setUpClass(cls):
        # Figure out where this is being run from and set the test data
        # path accordingly.
        cwd = os.getcwd()
        cls.test_data_path = "test-data"
        if cwd.endswith("python"):
            cls.test_data_path = "../test-data"

    def read_file(self, filename):
        full_path = os.path.join(self.test_data_path, filename)
        return kas.load(full_path, engine=self.engine, read_all=False)

    def test_empty_file(self):
        self.assertRaises(EOFError, self.read_file, "malformed/empty_file.kas")

    def test_bad_type(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_type_11.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_type_20.kas"
        )

    def test_bad_filesizes(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_0_-1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_0_1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_0_1024.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_10_-1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_10_1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_filesize_10_1024.kas"
        )

    def test_bad_magic_number(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_magic_number.kas"
        )

    def test_version_0(self):
        self.assertRaises(
            kas.VersionTooOldError, self.read_file, "malformed/version_0.kas"
        )

    def test_version_100(self):
        self.assertRaises(
            kas.VersionTooNewError, self.read_file, "malformed/version_100.kas"
        )

    def test_truncated_file(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/truncated_file.kas"
        )

    def test_key_offset_outside_file(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/key_offset_outside_file.kas"
        )

    def test_array_offset_outside_file(self):
        self.assertRaises(
            kas.FileFormatError,
            self.read_file,
            "malformed/array_offset_outside_file.kas",
        )

    def test_key_len_outside_file(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/key_len_outside_file.kas"
        )

    def test_array_len_outside_file(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/array_len_outside_file.kas"
        )

    def test_bad_array_start(self):
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_array_start_-8.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_array_start_-1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_array_start_1.kas"
        )
        self.assertRaises(
            kas.FileFormatError, self.read_file, "malformed/bad_array_start_8.kas"
        )

    def test_truncated_file_correct_size(self):
        self.assertRaises(
            kas.FileFormatError,
            self.read_file,
            "malformed/truncated_file_correct_size_100.kas",
        )
        self.assertRaises(
            kas.FileFormatError,
            self.read_file,
            "malformed/truncated_file_correct_size_128.kas",
        )
        self.assertRaises(
            kas.FileFormatError,
            self.read_file,
            "malformed/truncated_file_correct_size_129.kas",
        )
        self.assertRaises(
            kas.FileFormatError,
            self.read_file,
            "malformed/truncated_file_correct_size_200.kas",
        )

    def test_all_types(self):
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
            filename = f"v1/all_types_{n}_elements.kas"
            data = self.read_file(filename)
            for dtype in dtypes:
                self.assertTrue(np.array_equal(data[dtype], np.arange(n, dtype=dtype)))


class TestStandardFilesPyEngine(StandardFilesMixin, unittest.TestCase):
    engine = kas.PY_ENGINE
    read_all = False


class TestStandardFilesCEngine(StandardFilesMixin, unittest.TestCase):
    engine = kas.C_ENGINE
    read_all = False


class TestStandardFilesPyEngineReadAll(StandardFilesMixin, unittest.TestCase):
    engine = kas.PY_ENGINE
    read_all = True


class TestStandardFilesCEngineReadAll(StandardFilesMixin, unittest.TestCase):
    engine = kas.C_ENGINE
    read_all = True


class TestStandardFilesLoads(StandardFilesMixin, unittest.TestCase):
    def read_file(self, filename):
        full_path = os.path.join(self.test_data_path, filename)
        with open(full_path, "rb") as f:
            encoded = f.read()
        return kas.loads(encoded)
