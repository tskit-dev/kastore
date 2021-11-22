"""
Tests checking that the file format is as it should be.
"""
import os
import pathlib
import struct
import tempfile
import unittest

import hypothesis
import hypothesis.strategies as hst
import numpy as np

import kastore as kas
import kastore.store as store

# Set the deadline to None to avoid weird behaviour on CI.
hypothesis.settings.register_profile("kastore_defaults", deadline=None)
hypothesis.settings.load_profile("kastore_defaults")

# Exclude any 'other' unicode categories:
# http://www.unicode.org/reports/tr44/#General_Category_Values
key_alphabet = hst.characters(blacklist_categories=("C",))


class TestFileSignature(unittest.TestCase):
    """
    Checks the file signature is what we think it should be.
    """

    def test_form(self):
        self.assertEqual(len(store.MAGIC), 8)
        self.assertEqual(b"\211KAS\r\n\032\n", store.MAGIC)


class FormatMixin:
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
                {str(j): np.zeros(1) for j in range(n)},
                self.temp_file,
                engine=self.engine,
            )
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(contents[0:8], store.MAGIC)
            major, minor, num_items, size = struct.unpack("<HHIQ", contents[8:24])
            self.assertEqual(major, store.VERSION_MAJOR)
            self.assertEqual(minor, store.VERSION_MINOR)
            self.assertEqual(num_items, n)
            self.assertEqual(size, len(contents))
            trailer = contents[24 : store.HEADER_SIZE]
            # The remainder should be zeros.
            self.assertEqual(
                trailer, bytearray(0 for _ in range(store.HEADER_SIZE - 24))
            )

    def test_zero_items(self):
        kas.dump({}, self.temp_file, engine=self.engine)
        with open(self.temp_file, "rb") as f:
            contents = f.read()
        self.assertEqual(len(contents), 64)

    def test_item_descriptor_format(self):
        for n in range(10):
            kas.dump(
                {str(j): j * np.ones(j) for j in range(n)},
                self.temp_file,
                engine=self.engine,
            )
            with open(self.temp_file, "rb") as f:
                contents = f.read()
            self.assertEqual(struct.unpack("<I", contents[12:16])[0], n)
            offset = store.HEADER_SIZE
            for _ in range(n):
                descriptor = contents[offset : offset + store.ITEM_DESCRIPTOR_SIZE]
                offset += store.ITEM_DESCRIPTOR_SIZE
                type_ = struct.unpack("<B", descriptor[0:1])[0]
                key_start, key_len, array_start, array_len = struct.unpack(
                    "<QQQQ", descriptor[8:40]
                )
                trailer = descriptor[40 : store.ITEM_DESCRIPTOR_SIZE]
                # The remainder should be zeros.
                self.assertEqual(
                    trailer,
                    bytearray(0 for _ in range(store.ITEM_DESCRIPTOR_SIZE - 40)),
                )
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
                contents[offset : offset + store.ItemDescriptor.size]
            )
            descriptors.append(descriptor)
            offset += store.ItemDescriptor.size
        # Keys must be sorted lexicographically.
        sorted_keys = sorted(data.keys())
        # Keys should be packed sequentially immediately after the descriptors.
        offset = store.HEADER_SIZE + len(data) * store.ITEM_DESCRIPTOR_SIZE
        for d, key in zip(descriptors, sorted_keys):
            self.assertEqual(d.key_start, offset)
            unpacked_key = contents[d.key_start : d.key_start + d.key_len]
            self.assertEqual(key.encode("utf8"), unpacked_key)
            offset += d.key_len
        # Arrays should be packed sequentially immediately after the keys on
        # 8 byte boundaries
        for d, key in zip(descriptors, sorted_keys):
            remainder = offset % 8
            if remainder != 0:
                offset += 8 - remainder
            self.assertEqual(d.array_start, offset)
            nbytes = d.array_len * store.type_size(d.type)
            array = np.frombuffer(
                contents[d.array_start : d.array_start + nbytes],
                dtype=store.type_to_np_dtype_map[d.type],
            )
            np.testing.assert_equal(data[key], array)
            offset += nbytes

    def test_simple_key_storage(self):
        for n in range(10):
            self.validate_storage({"a" * (j + 1): np.ones(1) for j in range(n)})

    def test_simple_array_storage(self):
        for n in range(10):
            self.validate_storage({str(j): j * np.ones(j) for j in range(n)})

    # Note that hypothesis seems to be leaking memory, so when we're running tests
    # against the C API for memory leaks this must be commented out.
    @hypothesis.given(
        keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1)
    )
    def test_many_keys(self, keys):
        data = {key: np.ones(j, dtype=np.int32) * j for j, key in enumerate(keys)}
        self.validate_storage(data)


class TestFormatPyEngine(FormatMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


class TestFormatCEngine(FormatMixin, unittest.TestCase):
    engine = kas.C_ENGINE


class TestAlignedPacking(unittest.TestCase):
    """
    Tests that we are correctly computing alignments for the arrays.
    """

    def test_descriptor_str(self):
        items = {"a": np.array([1], dtype=np.int8)}
        descriptors, file_size = store.pack_items(items)
        self.assertGreater(len(str(descriptors[0])), 0)

    def test_single_key_1(self):
        items = {"a": np.array([1], dtype=np.int8)}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 128)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 136)
        self.assertEqual(d.array_len, 1)

    def test_single_key_7(self):
        items = {"aaaaaaa": np.array([1], dtype=np.int8)}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 128)
        self.assertEqual(d.key_len, 7)
        self.assertEqual(d.array_start, 136)
        self.assertEqual(d.array_len, 1)

    def test_single_key_8(self):
        items = {"aaaaaaaa": np.array([1], dtype=np.int8)}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 128)
        self.assertEqual(d.key_len, 8)
        self.assertEqual(d.array_start, 136)
        self.assertEqual(d.array_len, 1)

    def test_two_keys_array_len1(self):
        a = np.array([1], dtype=np.int8)
        items = {"a": a, "b": a}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 192)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 200)
        self.assertEqual(d.array_len, 1)
        d = descriptors[1]
        self.assertEqual(d.key_start, 193)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 208)
        self.assertEqual(d.array_len, 1)

    def test_two_keys_array_len8(self):
        a = np.array([1], dtype=np.int64)
        items = {"a": a, "b": a}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 192)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 200)
        self.assertEqual(d.array_len, 1)
        d = descriptors[1]
        self.assertEqual(d.key_start, 193)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 208)
        self.assertEqual(d.array_len, 1)

    def test_two_keys_array_len4(self):
        a = np.array([1], dtype=np.int32)
        items = {"a": a, "b": a}
        descriptors, file_size = store.pack_items(items)
        d = descriptors[0]
        self.assertEqual(d.key_start, 192)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 200)
        self.assertEqual(d.array_len, 1)
        d = descriptors[1]
        self.assertEqual(d.key_start, 193)
        self.assertEqual(d.key_len, 1)
        self.assertEqual(d.array_start, 208)
        self.assertEqual(d.array_len, 1)


class TestEnginesProduceIdenticalFiles(unittest.TestCase):
    """
    Ensure that the two engines produces identical files.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(suffix=".kas", prefix="kas_identity_test")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def verify(self, data):
        kas.dump(data, self.temp_file, engine=kas.C_ENGINE)
        with open(self.temp_file, "rb") as f:
            c_file = f.read()
        kas.dump(data, self.temp_file, engine=kas.PY_ENGINE)
        with open(self.temp_file, "rb") as f:
            py_file = f.read()
        self.assertEqual(c_file, py_file)

    def test_empty(self):
        self.verify({})

    def test_one_key_empty_array(self):
        self.verify({"a": []})

    def test_many_keys_empty_arrays(self):
        self.verify({"a" * (j + 1): [] for j in range(10)})

    def test_many_keys_nonempty_arrays(self):
        for dtype in [np.int8, np.uint8, np.uint32, np.float64]:
            self.verify({"a" * (j + 1): np.arange(j, dtype=dtype) for j in range(10)})

    def verify_all_dtypes(self, n):
        dtypes = [
            "int8",
            "uint8",
            "int16",
            "uint16",
            "uint32",
            "int32",
            "uint64",
            "int64",
            "float32",
            "float64",
        ]
        data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
        self.verify(data)

    def test_all_dtypes_0_elements(self):
        self.verify_all_dtypes(0)

    def test_all_dtypes_1_elements(self):
        self.verify_all_dtypes(1)

    def test_all_dtypes_2_elements(self):
        self.verify_all_dtypes(2)

    def test_all_dtypes_3_elements(self):
        self.verify_all_dtypes(3)

    def test_all_dtypes_4_elements(self):
        self.verify_all_dtypes(4)


class TruncatedFilesMixin:
    """
    Tests that we return the correct errors when we have truncated files.
    """

    def test_zero_bytes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = pathlib.Path(tempdir) / "testfile"
            with open(path, "wb") as f:
                f.write(b"")
            self.assertTrue(path.exists())
            with self.assertRaises(EOFError):
                kas.load(path, engine=self.engine)

    def test_short_file(self):
        example = {"a": np.arange(2)}
        encoded = kas.dumps(example)
        with tempfile.TemporaryDirectory() as tempdir:
            path = pathlib.Path(tempdir) / "testfile"
            for j in range(1, 64):  # len(encoded) - 2):
                with open(path, "wb") as f:
                    f.write(encoded[:j])
                self.assertTrue(path.exists())
                with self.assertRaises(kas.FileFormatError):
                    kas.load(path, engine=self.engine)


class TestTruncatedFilesPyEngine(unittest.TestCase, TruncatedFilesMixin):
    engine = kas.PY_ENGINE


class TestTruncatedFilesCEngine(unittest.TestCase, TruncatedFilesMixin):
    engine = kas.C_ENGINE
