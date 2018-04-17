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

import six
import numpy as np
import hypothesis
import hypothesis.strategies as hst

import kastore as kas
import kastore.store as store


# Exclude any 'other' unicode categories:
# http://www.unicode.org/reports/tr44/#General_Category_Values
key_alphabet = hst.characters(blacklist_categories=("C",))


class TestFileSignature(unittest.TestCase):
    """
    Checks the file signature is what we think it should be.
    """
    def test_form(self):
        self.assertEqual(len(store.MAGIC), 8)
        self.assertEqual(b'\211KAS\r\n\032\n', store.MAGIC)


class FormatMixin(object):
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
            kas.dump(
                {six.text_type(j): np.zeros(1) for j in range(n)}, self.temp_file,
                engine=self.engine)
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(contents[0:8], store.MAGIC)
            major, minor, num_items = struct.unpack("<HHI", contents[8:16])
            self.assertEqual(major, store.VERSION_MAJOR)
            self.assertEqual(minor, store.VERSION_MINOR)
            self.assertEqual(num_items, n)
            trailer = contents[16: store.HEADER_SIZE]
            # The remainder should be zeros.
            self.assertEqual(
                trailer, bytearray([0 for _ in range(store.HEADER_SIZE - 16)]))

    def test_zero_items(self):
        kas.dump({}, self.temp_file, engine=self.engine)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        self.assertEqual(len(contents), 64)

    def test_item_descriptor_format(self):
        for n in range(10):
            kas.dump(
                {six.text_type(j): j * np.ones(j) for j in range(n)}, self.temp_file,
                engine=self.engine)
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], n)
            offset = store.HEADER_SIZE
            for j in range(n):
                descriptor = contents[offset: offset + store.ITEM_DESCRIPTOR_SIZE]
                type_ = struct.unpack("<B", descriptor[0:1])[0]
                key_start, key_len, array_start, array_len = struct.unpack(
                    "<QQQQ", descriptor[8:40])
                trailer = descriptor[40: store.ITEM_DESCRIPTOR_SIZE]
                # The remainder should be zeros.
                self.assertEqual(
                    trailer,
                    bytearray([0 for _ in range(store.ITEM_DESCRIPTOR_SIZE - 40)]))
                self.assertEqual(descriptor[1:4], bytearray([0, 0, 0]))
                self.assertEqual(type_, store.FLOAT64)
                self.assertGreater(key_start, 0)
                self.assertGreater(key_len, 0)
                self.assertGreater(array_start, 0)
                self.assertGreaterEqual(array_len, 0)

    def validate_storage(self, data):
        kas.dump(data, self.temp_file, engine=self.engine)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        offset = store.HEADER_SIZE
        descriptors = []
        for _ in range(len(data)):
            descriptor = store.ItemDescriptor.unpack(
                contents[offset: offset + store.ItemDescriptor.size])
            descriptors.append(descriptor)
            offset += store.ItemDescriptor.size
        # Keys must be sorted lexicographically.
        sorted_keys = sorted(data.keys())
        # Keys should be packed sequentially immediately after the descriptors.
        offset = store.HEADER_SIZE + len(data) * store.ITEM_DESCRIPTOR_SIZE
        for d, key in zip(descriptors, sorted_keys):
            self.assertEqual(d.key_start, offset)
            unpacked_key = contents[d.key_start: d.key_start + d.key_len]
            self.assertEqual(key.encode("utf8"), unpacked_key)
            offset += d.key_len
        # Arrays should be packed sequentially immediately after the keys.
        for d, key in zip(descriptors, sorted_keys):
            self.assertEqual(d.array_start, offset)
            nbytes = d.array_len * store.type_size(d.type)
            array = np.frombuffer(
                contents[d.array_start: d.array_start + nbytes],
                dtype=store.type_to_np_dtype_map[d.type])
            np.testing.assert_equal(data[key], array)
            offset += nbytes

    def test_simple_key_storage(self):
        for n in range(10):
            self.validate_storage({"a" * (j + 1): np.ones(1) for j in range(n)})

    def test_simple_array_storage(self):
        for n in range(10):
            self.validate_storage({six.text_type(j): j * np.ones(j) for j in range(n)})

    # Note that hypothesis seems to be leaking memory, so when we're running tests
    # against the C API for memory leaks this must be commented out.
    @hypothesis.given(
        keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1))
    def test_many_keys(self, keys):
        data = {key: np.ones(j, dtype=np.int32) * j for j, key in enumerate(keys)}
        self.validate_storage(data)


class TestFormatPyEngine(FormatMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


class TestFormatCEngine(FormatMixin, unittest.TestCase):
    engine = kas.C_ENGINE
