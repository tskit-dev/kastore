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
import hypothesis
import hypothesis.strategies as hst

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
            self.assertEqual(struct.unpack("<H", contents[8:10])[0], kas.VERSION_MAJOR)
            self.assertEqual(struct.unpack("<H", contents[10:12])[0], kas.VERSION_MINOR)
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], n)
            trailer = contents[16: kas.HEADER_SIZE]
            # The remainder should be zeros.
            self.assertEqual(
                trailer, bytearray([0 for _ in range(kas.HEADER_SIZE - 16)]))

    def test_zero_items(self):
        with open(self.temp_file, "wb") as f:
            kas.dump({}, f)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        self.assertEqual(len(contents), 64)

    def test_item_descriptor_format(self):
        for n in range(10):
            with open(self.temp_file, "wb") as f:
                kas.dump({str(j): j * np.ones(j) for j in range(n)}, f)
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], n)
            offset = kas.HEADER_SIZE
            for j in range(n):
                descriptor = contents[offset: offset + kas.ITEM_DESCRIPTOR_SIZE]
                type_ = struct.unpack("<B", descriptor[0:1])[0]
                key_start = struct.unpack("<Q", descriptor[4:12])[0]
                key_len = struct.unpack("<Q", descriptor[12:20])[0]
                array_start = struct.unpack("<Q", descriptor[20:28])[0]
                array_len = struct.unpack("<Q", descriptor[28:36])[0]
                trailer = descriptor[36: kas.ITEM_DESCRIPTOR_SIZE]
                # The remainder should be zeros.
                self.assertEqual(
                    trailer,
                    bytearray([0 for _ in range(kas.ITEM_DESCRIPTOR_SIZE - 36)]))
                self.assertEqual(descriptor[1:4], bytearray([0, 0, 0]))
                self.assertEqual(type_, kas.FLOAT64)
                self.assertGreater(key_start, 0)
                self.assertGreater(key_len, 0)
                self.assertGreater(array_start, 0)
                self.assertGreaterEqual(array_len, 0)

    def validate_storage(self, data):
        with open(self.temp_file, "wb") as f:
            kas.dump(data, f)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        offset = kas.HEADER_SIZE
        descriptors = []
        for _ in range(len(data)):
            descriptor = kas.ItemDescriptor.unpack(
                contents[offset: offset + kas.ItemDescriptor.size])
            descriptors.append(descriptor)
            offset += kas.ItemDescriptor.size
        # Keys must be sorted lexicographically.
        sorted_keys = sorted(data.keys())
        # Keys should be packed sequentially immediately after the descriptors.
        offset = kas.HEADER_SIZE + len(data) * kas.ITEM_DESCRIPTOR_SIZE
        for d, key in zip(descriptors, sorted_keys):
            self.assertEqual(d.key_start, offset)
            unpacked_key = contents[d.key_start: d.key_start + d.key_len]
            self.assertEqual(key.encode("utf8"), unpacked_key)
            offset += d.key_len
        # Arrays should be packed sequentially immediately after the keys.
        for d, key in zip(descriptors, sorted_keys):
            self.assertEqual(d.array_start, offset)
            array = np.frombuffer(
                contents[d.array_start: d.array_start + d.array_len],
                dtype=kas.type_to_np_dtype_map[d.type])
            np.testing.assert_equal(data[key], array)
            offset += d.array_len

    def test_simple_key_storage(self):
        for n in range(10):
            self.validate_storage({"a" * (j + 1): np.ones(1) for j in range(n)})

    def test_simple_array_storage(self):
        for n in range(10):
            self.validate_storage({str(j): j * np.ones(j) for j in range(n)})

    @hypothesis.given(keys=hst.sets(hst.text(min_size=1), min_size=1))
    def test_many_keys(self, keys):
        data = {key: np.ones(j, dtype=np.int32) * j for j, key in enumerate(keys)}
        self.validate_storage(data)
