"""
The Python engine for kastore.

The file format layout is as follows.

+===================================+
+ Header (64 bytes)
+===================================+
+ Item descriptors (n * 64 bytes)
+===================================+
+ Keys densely.
+===================================+
+ Arrays packed densely.
+===================================+
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import struct
import logging

import numpy as np

logger = logging.getLogger(__name__)


# Magic number is derived from the strategy used by HDF5 and PNG;
# see https://support.hdfgroup.org/HDF5/doc/H5.format.html and
# http://www.libpng.org/pub/png/spec/iso/index-object.html#5PNG-file-signature.
# In ASCII C notation this is "\211KAS\r\n\032\n"
MAGIC = bytearray([137, 75, 65, 83, 13, 10, 26, 10])
HEADER_SIZE = 64
ITEM_DESCRIPTOR_SIZE = 64

VERSION_MAJOR = 0
VERSION_MINOR = 1

INT8 = 0
UINT8 = 1
INT32 = 2
UINT32 = 3
INT64 = 4
UINT64 = 5
FLOAT32 = 6
FLOAT64 = 7

np_dtype_to_type_map = {
    "int8": INT8,
    "uint8": UINT8,
    "uint32": UINT32,
    "int32": INT32,
    "uint64": UINT64,
    "int64": INT64,
    "float32": FLOAT32,
    "float64": FLOAT64,
}

type_to_np_dtype_map = {t: dtype for dtype, t in np_dtype_to_type_map.items()}


class ItemDescriptor(object):
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
            self, type_, key_start=None, key_len=None, array_start=None, array_len=None):
        self.type = type_
        self.key_start = key_start
        self.key_len = key_len
        self.array_start = array_start
        self.array_len = array_len
        self.key = None
        self.array = None

    def __str__(self):
        return "type={};key_start={};key_len={};array_start={};array_len={}".format(
                self.type, self.key_start, self.key_len, self.array_start,
                self.array_len)

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
    for key, array in arrays.items():
        if len(key) == 0:
            raise ValueError("Empty keys not supported")
        if len(array.shape) != 1:
            raise ValueError("Only 1D arrays supported")

    num_items = len(arrays)
    header_size = HEADER_SIZE
    header = bytearray(header_size)
    header[0:8] = MAGIC
    header[8:10] = struct.pack("<I", VERSION_MAJOR)
    header[10:12] = struct.pack("<H", VERSION_MINOR)
    header[12:16] = struct.pack("<H", num_items)
    # The rest of the header is reserved.
    fileobj.write(header)

    # We store the keys in sorted order in the key block.
    sorted_keys = sorted(arrays.keys())
    descriptor_block_size = num_items * ItemDescriptor.size
    offset = header_size + descriptor_block_size
    descriptors = []
    for key in sorted_keys:
        array = arrays[key]
        encoded_key = key.encode(key_encoding)
        descriptor = ItemDescriptor(np_dtype_to_type_map[array.dtype.name])
        descriptor.key = encoded_key
        descriptor.array = array
        descriptor.key_start = offset
        descriptor.key_len = len(encoded_key)
        offset += descriptor.key_len
        descriptors.append(descriptor)

    # Now pack the arrays in densely after the keys.
    for descriptor in descriptors:
        descriptor.array_start = offset
        descriptor.array_len = descriptor.array.nbytes
        offset += descriptor.array_len

    assert fileobj.tell() == header_size
    # Now write the descriptors.
    for descriptor in descriptors:
        fileobj.write(descriptor.pack())
    assert fileobj.tell() == header_size + descriptor_block_size
    # Write the keys and arrays
    for descriptor in descriptors:
        assert fileobj.tell() == descriptor.key_start
        fileobj.write(descriptor.key)
    for descriptor in descriptors:
        assert fileobj.tell() == descriptor.array_start
        fileobj.write(descriptor.array.data)


def load(fileobj, key_encoding="utf-8"):
    """
    Reads arrays from the specified file and returns the resulting mapping.
    """
    header_size = 64
    header = fileobj.read(header_size)
    if header[0:8] != MAGIC:
        raise ValueError("Incorrect file format")
    version_major = struct.unpack("<H", header[8:10])[0]
    version_minor = struct.unpack("<H", header[10:12])[0]
    logger.debug("Loading file version {}.{}".format(version_major, version_minor))
    if version_major != VERSION_MAJOR:
        raise ValueError("Incompatible major version")
    num_items = struct.unpack("<I", header[12:16])[0]
    logger.debug("Loading {} items".format(num_items))

    descriptor_block_size = num_items * ItemDescriptor.size
    descriptor_block = fileobj.read(descriptor_block_size)

    offset = 0
    descriptors = []
    for _ in range(num_items):
        descriptor = ItemDescriptor.unpack(
            descriptor_block[offset: offset + ItemDescriptor.size])
        descriptors.append(descriptor)
        offset += ItemDescriptor.size

    # Load the keys first, so that we do sequential IOs within the key block.
    for descriptor in descriptors:
        fileobj.seek(descriptor.key_start)
        assert fileobj.tell() == descriptor.key_start
        descriptor.key = fileobj.read(descriptor.key_len).decode(key_encoding)

    # Now load in the arrays.
    items = {}
    for descriptor in descriptors:
        items[descriptor.key] = descriptor.array
        fileobj.seek(descriptor.array_start)
        dtype = type_to_np_dtype_map[descriptor.type]
        data = fileobj.read(descriptor.array_len)
        descriptor.array = np.frombuffer(data, dtype=dtype)
        items[descriptor.key] = descriptor.array
        logger.debug("Loaded '{}'".format(descriptor.key))
    return items
