"""
Tests checking that the file format is as it should be.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import tempfile
import os
import struct

import numpy as np

import kastore as kas


class TestFileSignature(unittest.TestCase):
    """
    Checks the file signature is what we think it should be.
    """
    def test_form(self):
        self.assertEqual(len(kas.MAGIC), 8)
        self.assertEqual(b'\211KAS\r\n\032\n', kas.MAGIC)


class TestFormat(unittest.TestCase):
    """
    Tests for the file format.
    """
    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_rt_test")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_header_format(self):
        for n in range(10):
            kas.dump({str(j): np.zeros(1) for j in range(n)}, self.temp_file)
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(contents[0:8], kas.MAGIC)
            self.assertEqual(struct.unpack("<I", contents[8:12])[0], kas.VERSION_MAJOR)
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], kas.VERSION_MINOR)
            self.assertEqual(struct.unpack("<I", contents[16:20])[0], n)
