"""
Makes the files in the test-data directory.
"""
import os.path
import struct
import tempfile

import numpy as np

import kastore as kas
import kastore.store as store


class MalformedFilesBuilder:
    """
    Utility for making the malformed files.
    """

    def __init__(self):
        self.destination_dir = "../test-data/malformed"
        self.temp_file = tempfile.NamedTemporaryFile()

    def write_file(self, num_items=0):
        data = {}
        for j in range(num_items):
            data["a" * (j + 1)] = np.arange(j + 1, dtype=np.uint32)
        kas.dump(data, self.temp_file.name)

    def make_empty_file(self):
        filename = os.path.join(self.destination_dir, "empty_file.kas")
        with open(filename, "w"):
            pass

    def make_bad_item_types(self):
        items = {"a": []}
        descriptors, file_size = store.pack_items(items)
        num_types = len(store.np_dtype_to_type_map)
        for bad_type in [num_types + 1, 2 * num_types]:
            filename = os.path.join(self.destination_dir, f"bad_type_{bad_type}.kas")
            with open(filename, "wb") as f:
                descriptors[0].type = bad_type
                store.write_file(f, descriptors, file_size)

    def make_bad_file_sizes(self):
        for num_items in [0, 10]:
            for offset in [-1, 1, 2 ** 10]:
                self.write_file(num_items)
                file_size = os.path.getsize(self.temp_file.name)
                with open(self.temp_file.name, "rb") as f:
                    buff = bytearray(f.read())
                before_len = len(buff)
                buff[16:24] = struct.pack("<Q", file_size + offset)
                assert len(buff) == before_len

                filename = os.path.join(
                    self.destination_dir, f"bad_filesize_{num_items}_{offset}.kas"
                )
                with open(filename, "wb") as f:
                    f.write(buff)

    def make_bad_magic_number(self):
        self.write_file()
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        buff[0:8] = bytearray(0 for _ in range(8))
        filename = os.path.join(self.destination_dir, "bad_magic_number.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def write_version(self, version, filename):
        self.write_file()
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        buff[8:10] = struct.pack("<H", version)
        with open(filename, "wb") as f:
            f.write(buff)

    def make_version_0(self):
        filename = os.path.join(self.destination_dir, "version_0.kas")
        self.write_version(0, filename)

    def make_version_100(self):
        filename = os.path.join(self.destination_dir, "version_100.kas")
        self.write_version(100, filename)

    def make_truncated_file(self):
        self.write_file(10)
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        # 64 byte header + 1 descriptors is 128 bytes. Truncate at 150.
        buff = buff[:150]
        filename = os.path.join(self.destination_dir, "truncated_file.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def make_key_offset_outside_file(self):
        self.write_file(1)
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        off = 64
        buff[off + 8 : off + 16] = struct.pack("<Q", 2 ** 32)
        filename = os.path.join(self.destination_dir, "key_offset_outside_file.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def make_bad_key_start(self):
        for start_offset in [-1, 1]:
            self.write_file(1)
            with open(self.temp_file.name, "rb") as f:
                buff = bytearray(f.read())
            # The key should start at 128.
            off = 64
            buff[off + 8 : off + 16] = struct.pack("<Q", 128 + start_offset)
            filename = os.path.join(
                self.destination_dir, f"bad_key_start_{start_offset}.kas"
            )
            with open(filename, "wb") as f:
                f.write(buff)

    def make_array_offset_outside_file(self):
        self.write_file(1)
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        off = 64
        buff[off + 24 : off + 32] = struct.pack("<Q", 2 ** 32)
        filename = os.path.join(self.destination_dir, "array_offset_outside_file.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def make_bad_array_start(self):
        for start_offset in [-1, 1, -8, 8]:
            self.write_file(1)
            with open(self.temp_file.name, "rb") as f:
                buff = bytearray(f.read())
            # The array should start at 136.
            off = 64
            buff[off + 24 : off + 32] = struct.pack("<Q", 136 + start_offset)
            filename = os.path.join(
                self.destination_dir, f"bad_array_start_{start_offset}.kas"
            )
            with open(filename, "wb") as f:
                f.write(buff)

    def make_key_len_outside_file(self):
        self.write_file(1)
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        off = 64
        buff[off + 16 : off + 24] = struct.pack("<Q", 2 ** 32)
        filename = os.path.join(self.destination_dir, "key_len_outside_file.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def make_array_len_outside_file(self):
        self.write_file(1)
        with open(self.temp_file.name, "rb") as f:
            buff = bytearray(f.read())
        off = 64
        buff[off + 32 : off + 40] = struct.pack("<Q", 2 ** 32)
        filename = os.path.join(self.destination_dir, "array_len_outside_file.kas")
        with open(filename, "wb") as f:
            f.write(buff)

    def make_truncated_file_correct_filesize(self):
        # TODO make a bunch of files that have the correct file size in the
        # header, but his reflects a truncated file.
        for size in [100, 128, 129, 200]:
            self.write_file(5)
            with open(self.temp_file.name, "rb") as f:
                buff = bytearray(f.read())
            buff[16:24] = struct.pack("<Q", size)
            buff = buff[:size]
            filename = os.path.join(
                self.destination_dir, f"truncated_file_correct_size_{size}.kas"
            )
            with open(filename, "wb") as f:
                f.write(buff)

    def run(self):
        self.make_empty_file()
        self.make_bad_item_types()
        self.make_bad_file_sizes()
        self.make_bad_magic_number()
        self.make_version_0()
        self.make_version_100()
        self.make_truncated_file()
        self.make_key_offset_outside_file()
        self.make_array_offset_outside_file()
        self.make_key_len_outside_file()
        self.make_array_len_outside_file()
        self.make_bad_key_start()
        self.make_bad_array_start()
        self.make_truncated_file_correct_filesize()


def make_types_files():
    """
    Makes a set of files with 0 to 10 elements of each of the types.
    """
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
    destination_dir = "../test-data/v1"
    for n in range(10):
        data = {dtype: np.arange(n, dtype=dtype) for dtype in dtypes}
        filename = os.path.join(destination_dir, f"all_types_{n}_elements.kas")
        kas.dump(data, filename)


def main():
    mfb = MalformedFilesBuilder()
    mfb.run()
    make_types_files()


if __name__ == "__main__":
    main()
