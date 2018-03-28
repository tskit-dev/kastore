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
            with open(self.temp_file, "wb") as f:
                kas.dump({str(j): np.zeros(1) for j in range(n)}, f)
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(contents[0:8], kas.MAGIC)
            self.assertEqual(struct.unpack("<I", contents[8:12])[0], kas.VERSION_MAJOR)
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], kas.VERSION_MINOR)
            self.assertEqual(struct.unpack("<I", contents[16:20])[0], n)
            trailer = contents[20: kas.HEADER_SIZE]
            # The remainder should be zeros. Total length is 64, so 44 bytes remaining.
            self.assertEqual(trailer, bytes([0 for _ in range(kas.HEADER_SIZE - 20)]))

    def test_zero_items(self):
        with open(self.temp_file, "wb") as f:
            kas.dump({}, f)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        self.assertEqual(len(contents), 64)

    def test_item_descriptor_format(self):
        pass
