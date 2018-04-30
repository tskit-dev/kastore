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
+ Arrays packed densely.
+===================================+
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import struct
import logging
import mmap

import numpy as np
import six

from . import exceptions

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


def type_size(ka_type):
    """
    Returns the size in bytes of one item in the specified kastore type.
    """
    size_map = {
        INT8: 1,
        UINT8: 1,
        INT32: 4,
        UINT32: 4,
        INT64: 8,
        UINT64: 8,
        FLOAT32: 4,
        FLOAT64: 8,
    }
    return size_map[ka_type]


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


def dump(arrays, filename, key_encoding="utf-8"):
    with open(filename, "wb") as f:
        _dump(arrays, f, key_encoding)


def _dump(arrays, fileobj, key_encoding):
    """
    Writes the arrays in the specified mapping to the key-array-store file.
    """
    # This is really overly strict, but Python2 doesn't support
    # collections.abc.Mapping.
    if not isinstance(arrays, dict):
        raise TypeError("Input must be dict-like")
    for key, array in arrays.items():
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
        if not isinstance(key, six.text_type):
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
        remainder = (offset % ARRAY_ALIGN)
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
    fileobj.write(header)
    assert fileobj.tell() == HEADER_SIZE

    # Now write the descriptors.
    for descriptor in descriptors:
        fileobj.write(descriptor.pack())
    offset = HEADER_SIZE + num_items * ItemDescriptor.size
    assert fileobj.tell() == offset
    # Write the keys and arrays
    for descriptor in descriptors:
        fileobj.write(descriptor.key)
        offset += descriptor.key_len
    assert fileobj.tell() == offset
    for descriptor in descriptors:
        # Because of the alignment requirements for array storage we
        # need to add in some padding between arrays.
        padding = descriptor.array_start - offset
        fileobj.write(b'\0' * padding)
        data = bytes(descriptor.array.data)
        fileobj.write(data)
        offset = descriptor.array_start + len(data)


def load(filename, use_mmap=True, key_encoding="utf-8"):
    with open(filename, "rb") as f:
        return _load(f, key_encoding)


def _load(fileobj, key_encoding):
    """
    Reads arrays from the specified file and returns the resulting mapping.
    """
    header = fileobj.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise exceptions.FileFormatError("Truncated header")
    if header[0:8] != MAGIC:
        raise exceptions.FileFormatError("Magic number mismatch")
    version_major = struct.unpack("<H", header[8:10])[0]

    version_minor = struct.unpack("<H", header[10:12])[0]
    logger.debug("Loading file version {}.{}".format(version_major, version_minor))
    if version_major < VERSION_MAJOR:
        raise exceptions.VersionTooOldError()
    elif version_major > VERSION_MAJOR:
        raise exceptions.VersionTooNewError()
    num_items, file_size = struct.unpack("<IQ", header[12:24])
    logger.debug("Loading {} items from {} bytes".format(num_items, file_size))

    descriptor_block_size = num_items * ItemDescriptor.size
    descriptor_block = fileobj.read(descriptor_block_size)
    if len(descriptor_block) != descriptor_block_size:
        raise exceptions.FileFormatError("Truncated file")

    offset = 0
    descriptors = []
    for _ in range(num_items):
        descriptor = ItemDescriptor.unpack(
            descriptor_block[offset: offset + ItemDescriptor.size])
        descriptors.append(descriptor)
        offset += ItemDescriptor.size
        # Check the type.
        num_types = len(np_dtype_to_type_map)
        if descriptor.type >= num_types:
            raise exceptions.FileFormatError("Unknown type")
        # Check the descriptor addresses are within the file.
        if descriptor.key_start + descriptor.key_len > file_size:
            raise exceptions.FileFormatError("Key address outside file")
        array_end = (
            descriptor.array_start +
            type_size(descriptor.type) * descriptor.array_len)
        if array_end > file_size:
            raise exceptions.FileFormatError("Array address outside file")

    offset += HEADER_SIZE
    # Load the keys first, so that we do sequential IOs within the key block.
    for descriptor in descriptors:
        if descriptor.key_start != offset:
            raise exceptions.FileFormatError("Keys not packed sequentially")
        fileobj.seek(descriptor.key_start)
        descriptor.key = fileobj.read(descriptor.key_len).decode(key_encoding)
        offset += descriptor.key_len

    # Now load in the arrays.
    items = {}
    for descriptor in descriptors:
        items[descriptor.key] = descriptor.array
        remainder = offset % ARRAY_ALIGN
        if remainder != 0:
            offset += ARRAY_ALIGN - remainder
        if descriptor.array_start % ARRAY_ALIGN != 0:
            raise exceptions.FileFormatError("Arrays must be 8 byte aligned")
        if descriptor.array_start != offset:
            raise exceptions.FileFormatError(
                "Arrays must sequentially packed and 8 byte aligned")
        fileobj.seek(descriptor.array_start)
        try:
            dtype = type_to_np_dtype_map[descriptor.type]
        except KeyError:
            raise exceptions.FileFormatError("Unknown data type")
        data = fileobj.read(descriptor.array_len * type_size(descriptor.type))
        descriptor.array = np.frombuffer(data, dtype=dtype)
        items[descriptor.key] = descriptor.array
        logger.debug("Loaded '{}'".format(descriptor.key))
        offset += descriptor.array_len * type_size(descriptor.type)

    if fileobj.tell() != file_size:
        raise exceptions.FileFormatError("Bad file size in header")
    return items


MODE_READ = 1
MODE_WRITE = 2


class Store(object):

    def __init__(self, filename, mode, key_encoding="utf8"):
        self.filename = filename
        self.key_encoding = key_encoding
        self._file = None
        if mode == 'r':
            self.mode = MODE_READ
            self._open_read()
        elif mode == 'w':
            self.mode = MODE_WRITE
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    def _open_read(self):
        self.file = open(self.filename, "rb")
        header = self.file.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise exceptions.FileFormatError("Truncated header")
        if header[0:8] != MAGIC:
            raise exceptions.FileFormatError("Magic number mismatch")
        version_major = struct.unpack("<H", header[8:10])[0]

        version_minor = struct.unpack("<H", header[10:12])[0]
        logger.debug("Loading file version {}.{}".format(version_major, version_minor))
        if version_major < VERSION_MAJOR:
            raise exceptions.VersionTooOldError()
        elif version_major > VERSION_MAJOR:
            raise exceptions.VersionTooNewError()
        num_items, file_size = struct.unpack("<IQ", header[12:24])
        logger.debug("Loading {} items from {} bytes".format(num_items, file_size))

        self._mmap = mmap.mmap(self.file.fileno(), file_size, prot=mmap.PROT_READ)

        descriptor_block_size = num_items * ItemDescriptor.size
        descriptor_block = self._mmap[HEADER_SIZE: HEADER_SIZE + descriptor_block_size]

        if len(descriptor_block) != descriptor_block_size:
            raise exceptions.FileFormatError("Truncated file")

        offset = 0
        descriptors = []
        for _ in range(num_items):
            descriptor = ItemDescriptor.unpack(
                descriptor_block[offset: offset + ItemDescriptor.size])
            descriptors.append(descriptor)
            offset += ItemDescriptor.size
            # Check the type.
            num_types = len(np_dtype_to_type_map)
            if descriptor.type >= num_types:
                raise exceptions.FileFormatError("Unknown type")
            # Check the descriptor addresses are within the file.
            if descriptor.key_start + descriptor.key_len > file_size:
                raise exceptions.FileFormatError("Key address outside file")
            array_end = (
                descriptor.array_start +
                type_size(descriptor.type) * descriptor.array_len)
            if array_end > file_size:
                raise exceptions.FileFormatError("Array address outside file")

        # Read the keys.
        offset = HEADER_SIZE + descriptor_block_size
        for descriptor in descriptors:
            if descriptor.key_start != offset:
                raise exceptions.FileFormatError("Keys not packed sequentially")
            key = self._mmap[offset: offset + descriptor.key_len]
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
                    "Arrays must sequentially packed and 8 byte aligned")
            if descriptor.type not in type_to_np_dtype_map:
                raise exceptions.FileFormatError("Unknown data type")
            offset += descriptor.array_len * type_size(descriptor.type)

        # Create the mapping for descriptors.
        self._descriptor_map = {descriptor.key: descriptor for descriptor in descriptors}

    def __getitem__(self, key):
        descriptor = self._descriptor_map[key]
        size = type_size(descriptor.type) * descriptor.array_len
        data = self._mmap[descriptor.array_start: descriptor.array_start + size]
        dtype = type_to_np_dtype_map[descriptor.type]
        array = np.frombuffer(data, dtype=dtype)
        logger.debug("Loaded '{}'".format(descriptor.key))
        return array
