"""
Tests for error conditions.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import tempfile
import os
import platform
import sys

import numpy as np

import kastore as kas

IS_WINDOWS = platform.system() == "Windows"
IS_PY2 = sys.version_info[0] < 3


class InterfaceMixin(object):
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
            self.assertRaises(
                TypeError, kas.dump, bad_dict, self.temp_file, engine=self.engine)
            self.assertRaises(
                TypeError, kas.dump, data=bad_dict, filename=self.temp_file,
                engine=self.engine)

    def test_bad_filename_type(self):
        for bad_filename in [[], None, {}]:
            self.assertRaises(
                TypeError, kas.dump, {}, bad_filename, engine=self.engine)
            self.assertRaises(
                TypeError, kas.dump, data={}, filename=bad_filename, engine=self.engine)
            self.assertRaises(
                TypeError, kas.load, bad_filename, engine=self.engine)
            self.assertRaises(
                TypeError, kas.load, filename=bad_filename, engine=self.engine)

    def test_bad_keys(self):
        a = np.zeros(1)
        for bad_key in [(1234,), b"1234", None, 1234]:
            self.assertRaises(
                TypeError, kas.dump, data={bad_key: a}, filename=self.temp_file,
                engine=self.engine)

    def test_bad_arrays(self):
        kas.dump(data={"a": []}, filename=self.temp_file, engine=self.engine)
        for bad_array in [kas, lambda x: x, "1234", None, [[0, 1], [0, 2]]]:
            self.assertRaises(
                ValueError, kas.dump, data={"a": bad_array},
                filename=self.temp_file, engine=self.engine)
        # TODO add tests for arrays in fortran order and so on.

    @unittest.skipIf(IS_PY2, "Skip IO errors for py2")
    def test_file_not_found(self):
        a = np.zeros(1)
        for bad_file in ["no_such_file", "/no/such/file"]:
            self.assertRaises(
                FileNotFoundError, kas.load, filename=bad_file, engine=self.engine)
        self.assertRaises(
            FileNotFoundError, kas.dump, data={"a": a}, filename="/no/such/file",
            engine=self.engine)

    @unittest.skipIf(IS_PY2, "Skip IO errors for py2")
    def test_file_is_a_directory(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            exception = IsADirectoryError
            if IS_WINDOWS:
                exception = PermissionError
            self.assertRaises(
                exception, kas.dump, filename=tmp_dir, data={"a": []},
                engine=self.engine)
            # TODO fix this when low-level issues resolved.
            # self.assertRaises(
            #     exception, kas.load, filename=tmp_dir, engine=self.engine)
        finally:
            os.rmdir(tmp_dir)


class TestInterfacePyEngine(InterfaceMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


class TestInterfaceCEngine(InterfaceMixin, unittest.TestCase):
    engine = kas.C_ENGINE


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
