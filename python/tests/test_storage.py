import os
import tempfile
import unittest

import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hst
import numpy as np

import kastore as kas


"""
Basic tests for the storage integrity of the data.
"""


# Set the deadline to None to avoid weird behaviour on CI.
hypothesis.settings.register_profile(
    "kastore_defaults",
    deadline=None,
    # Supress warnings resultsing from inheritance
    suppress_health_check=(hypothesis.HealthCheck.differing_executors,),
)
hypothesis.settings.load_profile("kastore_defaults")

# Exclude any 'other' unicode categories:
# http://www.unicode.org/reports/tr44/#General_Category_Values
key_alphabet = hst.characters(blacklist_categories=("C",))


class TestInputErrors(unittest.TestCase):
    """
    Simple input errors.
    """

    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_input")
        os.close(fd)
        self.temp_file = path

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_empty_key(self):
        self.assertRaises(ValueError, kas.dump, {"": np.zeros(1)}, self.temp_file)

    def test_2d_array(self):
        self.assertRaises(ValueError, kas.dump, {"a": np.zeros((1, 1))}, self.temp_file)

    def test_3d_array(self):
        self.assertRaises(
            ValueError, kas.dump, {"a": np.zeros((2, 2, 1))}, self.temp_file
        )


class TestRoundTrip(unittest.TestCase):
    """
    Tests that we can round trip data through a temporary file.
    """

    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_rt")
        os.close(fd)
        self.temp_file = path

    def tearDown(self):
        os.unlink(self.temp_file)

    def verify(self, data):
        for engine in [kas.C_ENGINE, kas.PY_ENGINE]:
            for read_all in [True, False]:
                kas.dump(data, self.temp_file, engine=engine)
                new_data = kas.load(self.temp_file, read_all=read_all, engine=engine)
                self.assertEqual(sorted(new_data.keys()), sorted(data.keys()))
                for key, source_array in data.items():
                    dest_array = new_data[key]
                    # Numpy's testing assert_equal will deal correctly with NaNs.
                    np.testing.assert_equal(source_array, dest_array)
                # Make sure the file is closed before opening it again.
                del new_data


class BytesRoundTripMixin:
    """
    A mixin to which can be used to inherit the same tests defined for subclasses
    of TestRoundTrip. Note this *must* be the first class inherited from.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def verify(self, data):
        encoded = kas.dumps(data)
        new_data = kas.loads(encoded)
        self.assertEqual(sorted(new_data.keys()), sorted(data.keys()))
        for key, source_array in data.items():
            dest_array = new_data[key]
            # Numpy's testing assert_equal will deal correctly with NaNs.
            np.testing.assert_equal(source_array, dest_array)


class TestRoundTripSimple(TestRoundTrip):
    """
    Simple round-trip tests for some hand crafted cases.
    """

    def test_zero_keys(self):
        self.verify({})

    def test_single_key(self):
        self.verify({"a": np.zeros(1)})

    def test_many_keys(self):
        data = {}
        for j in range(1):
            data[str(j)] = j + np.zeros(j, dtype=np.uint32)
        self.verify(data)

    def test_all_dtypes(self):
        dtypes = [
            "int8",
            "uint8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "uint64",
            "int64",
            "float32",
            "float64",
        ]
        for n in range(10):
            data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
            self.verify(data)


class TestRoundTripSimpleBytes(BytesRoundTripMixin, TestRoundTripSimple):
    pass


class TestRoundTripKeys(TestRoundTrip):
    """
    Test round tripping with keys generated by hypothesis.
    """

    @hypothesis.given(key=hst.text(alphabet=key_alphabet, min_size=1))
    def test_single_key(self, key):
        self.verify({key: np.zeros(1)})

    @hypothesis.given(
        keys=hst.sets(hst.text(alphabet=key_alphabet, min_size=1), min_size=1)
    )
    def test_many_keys(self, keys):
        data = {key: np.ones(j) * j for j, key in enumerate(keys)}
        self.verify(data)


class TestRoundTripKeysBytes(BytesRoundTripMixin, TestRoundTripKeys):
    pass


shape_strategy = hnp.array_shapes(max_dims=1)


class TestRoundTripDataTypes(TestRoundTrip):
    """
    Test round tripping of the various types using Hypothesis.
    """

    @hypothesis.given(value=hnp.arrays(dtype=np.uint8, shape=shape_strategy))
    def test_single_uint8(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.int8, shape=shape_strategy))
    def test_single_int8(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.int16, shape=shape_strategy))
    def test_single_int16(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.uint16, shape=shape_strategy))
    def test_single_uint16(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.int32, shape=shape_strategy))
    def test_single_int32(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.uint32, shape=shape_strategy))
    def test_single_uint32(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.int64, shape=shape_strategy))
    def test_single_int64(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.uint64, shape=shape_strategy))
    def test_single_uint64(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.float32, shape=shape_strategy))
    def test_single_float32(self, value):
        self.verify({"a": value})

    @hypothesis.given(value=hnp.arrays(dtype=np.float64, shape=shape_strategy))
    def test_single_float64(self, value):
        self.verify({"a": value})


class TestRoundTripDataTypesBytes(BytesRoundTripMixin, TestRoundTripDataTypes):
    pass
