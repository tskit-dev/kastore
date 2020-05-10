"""
The Python engine for kastore.

The file format layout is as follows.

+===================================+
+ Header (64 bytes)
+===================================+
+ Item descriptors (n * 64 bytes)
+===================================+
+ Keys packed densely.
+===================================+
+ Arrays packed densely starting on
+ 8 byte bounaries.
+===================================+
"""
import logging
import os
import struct
from collections.abc import Mapping

import numpy as np

import kastore.exceptions as exceptions

logger = logging.getLogger(__name__)


# Magic number is derived from the strategy used by HDF5 and PNG;
# see https://support.hdfgroup.org/HDF5/doc/H5.format.html and
# http://www.libpng.org/pub/png/spec/iso/index-object.html#5PNG-file-signature.
# In ASCII C notation this is "\211KAS\r\n\032\n"
MAGIC = bytearray([137, 75, 65, 83, 13, 10, 26, 10])
HEADER_SIZE = 64
ITEM_DESCRIPTOR_SIZE = 64
# Arrays must be stored with the start aligned on 8 byte boundaries to
# allow for aligned access when using mmap.
ARRAY_ALIGN = 8

VERSION_MAJOR = 1
VERSION_MINOR = 0

INT8 = 0
UINT8 = 1
INT16 = 2
UINT16 = 3
INT32 = 4
UINT32 = 5
INT64 = 6
UINT64 = 7
FLOAT32 = 8
FLOAT64 = 9

np_dtype_to_type_map = {
    "int8": INT8,
    "uint8": UINT8,
    "int16": INT16,
    "uint16": UINT16,
    "uint32": UINT32,
    "int32": INT32,
    "uint64": UINT64,
    "int64": INT64,
    "float32": FLOAT32,
    "float64": FLOAT64,
}

type_to_np_dtype_map = {t: dtype for dtype, t in np_dtype_to_type_map.items()}


def type_size(ka_type):
    """
    Returns the size in bytes of one item in the specified kastore type.
    """
    size_map = {
        INT8: 1,
        UINT8: 1,
        INT16: 2,
        UINT16: 2,
        INT32: 4,
        UINT32: 4,
        INT64: 8,
        UINT64: 8,
        FLOAT32: 4,
        FLOAT64: 8,
    }
    return size_map[ka_type]


class ItemDescriptor:
    """
    The information required to recover a single key-value pair from the
    file. Each descriptor is a block of 64 bytes, which stores:

    - The numeric type of the array (similar to numpy dtype)
    - The start offset of the key
    - The length of the key
    - The start offset of the array
    - The length of the array

    File offsets are stored as 8 byte unsigned little endian integers.
    The remaining space in the descriptor is reserved for later use.
    For example, we may wish to add an 'encoding' field in the future,
    which allows for things like simple run-length encoding and so on.
    """

    size = ITEM_DESCRIPTOR_SIZE

    def __init__(
        self, type_, key_start=None, key_len=None, array_start=None, array_len=None
    ):
        self.type = type_
        self.key_start = key_start
        self.key_len = key_len
        self.array_start = array_start
        self.array_len = array_len
        self.key = None
        self.array = None

    def __str__(self):
        return "type={};key_start={};key_len={};array_start={};array_len={}".format(
            self.type, self.key_start, self.key_len, self.array_start, self.array_len
        )

    def pack(self):
        descriptor = bytearray(ITEM_DESCRIPTOR_SIZE)
        descriptor[0:1] = struct.pack("<B", self.type)
        # bytes 1:8 are reserved.
        descriptor[8:16] = struct.pack("<Q", self.key_start)
        descriptor[16:24] = struct.pack("<Q", self.key_len)
        descriptor[24:32] = struct.pack("<Q", self.array_start)
        descriptor[32:40] = struct.pack("<Q", self.array_len)
        # bytes 40:64 are reserved.
        return descriptor

    @classmethod
    def unpack(cls, descriptor):
        type_ = struct.unpack("<B", descriptor[0:1])[0]
        key_start = struct.unpack("<Q", descriptor[8:16])[0]
        key_len = struct.unpack("<Q", descriptor[16:24])[0]
        array_start = struct.unpack("<Q", descriptor[24:32])[0]
        array_len = struct.unpack("<Q", descriptor[32:40])[0]
        return cls(type_, key_start, key_len, array_start, array_len)


def dump(arrays, fileobj, key_encoding="utf-8"):
    """
    Writes the arrays in the specified mapping to the key-array-store file.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("Input must be dict-like")
    for key in arrays.keys():
        if len(key) == 0:
            raise ValueError("Empty keys not supported")
    descriptors, file_size = pack_items(arrays, key_encoding)
    write_file(fileobj, descriptors, file_size)


def pack_items(arrays, key_encoding="utf-8"):
    """
    Packs the specified items by computing the relevant file offsets
    and return the list of ItemDescriptors and the overall size of the
    file.
    """
    num_items = len(arrays)
    # We store the keys in sorted order in the key block.
    sorted_keys = sorted(arrays.keys())
    descriptor_block_size = num_items * ItemDescriptor.size
    # First pack the keys and arrays to determine their locations.
    offset = HEADER_SIZE + descriptor_block_size
    descriptors = []
    for key in sorted_keys:
        # Normally we wouldn't bother with this in Python, but we want to
        # ensure that the behaviour is identical to the low-level module.
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        array = np.array(arrays[key])
        if len(array.shape) != 1:
            raise ValueError("Only 1D arrays supported")
        encoded_key = key.encode(key_encoding)
        descriptor = ItemDescriptor(np_dtype_to_type_map[str(array.dtype)])
        descriptor.key = encoded_key
        descriptor.array = array
        descriptor.key_start = offset
        descriptor.key_len = len(encoded_key)
        offset += descriptor.key_len
        descriptors.append(descriptor)
    # Now pack the arrays in densely after the keys 8 byte aligned
    for descriptor in descriptors:
        remainder = offset % ARRAY_ALIGN
        if remainder != 0:
            offset += ARRAY_ALIGN - remainder
        descriptor.array_start = offset
        descriptor.array_len = descriptor.array.shape[0]
        offset += descriptor.array_len * type_size(descriptor.type)
    return descriptors, offset


def write_file(fileobj, descriptors, file_size):
    """
    Writes the specified list of ItemDescriptors defining the file
    packing to the specified opened file object.
    """
    header = bytearray(HEADER_SIZE)
    num_items = len(descriptors)
    header[0:8] = MAGIC
    header[8:10] = struct.pack("<I", VERSION_MAJOR)
    header[10:12] = struct.pack("<H", VERSION_MINOR)
    header[12:16] = struct.pack("<H", num_items)
    header[16:24] = struct.pack("<Q", file_size)
    # The rest of the header is reserved.
    nbytes = fileobj.write(header)
    assert nbytes == HEADER_SIZE

    # Now write the descriptors.
    for descriptor in descriptors:
        nbytes += fileobj.write(descriptor.pack())
    offset = HEADER_SIZE + num_items * ItemDescriptor.size
    assert nbytes == offset
    # Write the keys and arrays
    for descriptor in descriptors:
        nbytes += fileobj.write(descriptor.key)
        offset += descriptor.key_len
    assert nbytes == offset
    for descriptor in descriptors:
        # Because of the alignment requirements for array storage we
        # need to add in some padding between arrays.
        padding = descriptor.array_start - offset
        fileobj.write(b"\0" * padding)
        data = bytes(descriptor.array.data)
        fileobj.write(data)
        offset = descriptor.array_start + len(data)


def load(file, read_all=False, key_encoding="utf-8"):
    return Store(file, read_all=read_all, key_encoding=key_encoding)


class ValueInfo:
    """
    Simple class encapsulating information about a store array.
    """

    def __init__(self, dtype, shape, size):
        self.dtype = dtype
        self.shape = shape
        self.size = size

    def __str__(self):
        return f"dtype={self.dtype} shape={self.shape}, size={self.size}"


class Store(Mapping):
    """
    Dictionary-like object giving dynamic access to the data in a kastore.
    """

    def __init__(self, file, read_all=False, key_encoding="utf8"):

        # Get ourselves a local version of the file. The semantics here are complex
        # because need to support a range of inputs and the free behaviour is
        # slightly different on each.
        self._file = None
        self._local_file = True
        try:
            # First, see if we can interpret the argument as a pathlike object.
            path = os.fspath(file)
            self._file = open(path, "rb")
        except TypeError:
            pass
        if self._file is None:
            # Now we try to open file. If it's not a pathlike object, it could be
            # an integer fd or object with a fileno method. In this case we
            # must make sure that close is **not** called on the fd.
            try:
                self._file = open(file, "rb", closefd=False, buffering=0)
            except TypeError:
                pass
        if self._file is None:
            # Assume that this is a file **but** we haven't opened it, so we must
            # not close it.
            if not hasattr(file, "write"):
                raise TypeError("file object must have a write method")
            self._file = file
            self._local_file = False

        self.key_encoding = key_encoding
        self._descriptor_map = None
        self._array_map = {}
        self._read_all = read_all
        self._file_offset = 0
        if not read_all:
            # Record the current file offset, in case this is a multi-store file,
            # so that we can seek to the correct location in __getitem__().
            self._file_offset = self._file.tell()
        self._read_file()

    def _read_header(self):
        first_byte = self._file.read(1)
        if len(first_byte) != 1:
            raise EOFError("End of file")

        header = first_byte + self._read(HEADER_SIZE - 1)
        if header[0:8] != MAGIC:
            raise exceptions.FileFormatError("Magic number mismatch")
        version_major = struct.unpack("<H", header[8:10])[0]
        version_minor = struct.unpack("<H", header[10:12])[0]
        logger.debug(f"Loading file version {version_major}.{version_minor}")
        if version_major < VERSION_MAJOR:
            raise exceptions.VersionTooOldError()
        elif version_major > VERSION_MAJOR:
            raise exceptions.VersionTooNewError()
        num_items, file_size = struct.unpack("<IQ", header[12:24])
        logger.debug(f"Loading {num_items} items from {file_size} bytes")
        if file_size < HEADER_SIZE:
            raise exceptions.FileFormatError("Bad file size in header")
        return num_items, file_size

    def _read_descriptors(self, num_items, file_size):
        descriptor_block_size = num_items * ItemDescriptor.size
        if HEADER_SIZE + descriptor_block_size > file_size:
            raise exceptions.FileFormatError("Bad file size in header")

        descriptor_block = self._read(descriptor_block_size)
        offset = 0
        array_end = HEADER_SIZE
        descriptors = []
        for _ in range(num_items):
            descriptor = ItemDescriptor.unpack(
                descriptor_block[offset : offset + ItemDescriptor.size]
            )
            descriptors.append(descriptor)
            offset += ItemDescriptor.size
            # Check the type.
            num_types = len(np_dtype_to_type_map)
            if descriptor.type >= num_types:
                raise exceptions.FileFormatError("Unknown type")
            # Check the descriptor addresses are within the file.
            if descriptor.key_start + descriptor.key_len > file_size:
                raise exceptions.FileFormatError("Key address outside file")
            if descriptor.array_start < HEADER_SIZE + descriptor_block_size:
                raise exceptions.FileFormatError("Array address out of bounds")
            array_end = (
                descriptor.array_start
                + type_size(descriptor.type) * descriptor.array_len
            )
            if array_end > file_size:
                raise exceptions.FileFormatError("Array address outside file")
        if array_end != file_size:
            raise exceptions.FileFormatError("Bad file size in header")
        return descriptors

    def _read_file(self):
        num_items, file_size = self._read_header()
        descriptors = self._read_descriptors(num_items, file_size)

        if num_items > 0:
            offset = HEADER_SIZE + num_items * ItemDescriptor.size
            if self._read_all:
                size = file_size
            else:
                # Read only the keys.
                size = descriptors[0].array_start
            assert size > offset
            size -= offset
            buf = self._read(size)
            buf_start = offset  # buffer starts at this position in the file

            # Read the keys.
            for descriptor in descriptors:
                if descriptor.key_start != offset:
                    raise exceptions.FileFormatError("Keys not packed sequentially")
                buf_offset = descriptor.key_start - buf_start
                key = buf[buf_offset : buf_offset + descriptor.key_len]
                descriptor.key = key.decode(self.key_encoding)
                offset += descriptor.key_len

            # Check the arrays.
            for descriptor in descriptors:
                remainder = offset % ARRAY_ALIGN
                if remainder != 0:
                    offset += ARRAY_ALIGN - remainder
                if descriptor.array_start % ARRAY_ALIGN != 0:
                    raise exceptions.FileFormatError("Arrays must be 8 byte aligned")
                if descriptor.array_start != offset:
                    raise exceptions.FileFormatError(
                        "Arrays must be sequentially packed and 8 byte aligned"
                    )
                offset += descriptor.array_len * type_size(descriptor.type)

        # Create the mapping for descriptors.
        self._descriptor_map = {
            descriptor.key: descriptor for descriptor in descriptors
        }

        if self._read_all:
            # Get the arrays from the buffer.
            for descriptor in descriptors:
                buf_offset = descriptor.array_start - buf_start
                size = type_size(descriptor.type) * descriptor.array_len
                data = buf[buf_offset : buf_offset + size]
                self._cache_array(descriptor, data)

    def _cache_array(self, descriptor, data):
        dtype = type_to_np_dtype_map[descriptor.type]
        array = np.frombuffer(data, dtype=dtype)
        self._array_map[descriptor.key] = array
        logger.debug(f"Loaded '{descriptor.key}'")

    def _check_open(self):
        if self._file is None:
            raise exceptions.StoreClosedError()

    def _read(self, size):
        """
        Reads exactly size bytes from the file and returns them.
        """
        data = b""
        while len(data) < size:
            chunk = self._file.read(size - len(data))
            data += chunk
            if len(chunk) == 0:  # EOF
                raise exceptions.FileFormatError("Truncated file")
        return data

    def info(self, key):
        self._check_open()
        descriptor = self._descriptor_map[key]
        dtype = type_to_np_dtype_map[descriptor.type]
        size = type_size(descriptor.type) * descriptor.array_len
        shape = (descriptor.array_len,)
        return ValueInfo(dtype, shape, size)

    def __getitem__(self, key):
        self._check_open()
        if key not in self._array_map:
            descriptor = self._descriptor_map[key]
            self._file.seek(self._file_offset + descriptor.array_start)
            size = type_size(descriptor.type) * descriptor.array_len
            data = self._read(size)
            self._cache_array(descriptor, data)
        return self._array_map[key]

    def __len__(self):
        self._check_open()
        return len(self._descriptor_map)

    def __iter__(self):
        self._check_open()
        yield from self._descriptor_map.keys()

    def close(self):
        if self._file is not None:
            if self._local_file:
                self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
