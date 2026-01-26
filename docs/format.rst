.. _sec_format:

===========
File format
===========

Binary File Structure
---------------------

Each file consists of three sections stored sequentially in binary on disk: the **Header**, **Keys Table**, and **Data Blocks**. These sections are aligned to facilitate platform-independent access and ensure compatibility across architectures.

### 1. Header Section

The **Header** occupies a fixed length and appears at the beginning of the file. It acts as a preamble and provides all necessary metadata for identifying the file, navigating internal structures, and verifying compatibility.

- **Magic Number (4 bytes)**: A file identifier (`b'KAST'`) that uniquely marks the file as being in kastore format.
- **File Version (4 bytes)**: The major and minor version numbers, ensuring forward and backward compatibility.
    - **Major Version (2 bytes)**: Incremented when breaking changes are introduced.
    - **Minor Version (2 bytes)**: Incremented when non-breaking, backward-compatible changes are made.
- **Flags (4 bytes)**: Feature-specific flags used during interpretation.
- **Num Keys (4 bytes)**: Number of key-value pairs stored in this file.
- **Table Offsets (16 bytes)**:
    - Offset to the **Keys Table**.
    - Offset to the **Data Blocks**.
- **Reserved (4 bytes)**: Reserved for future extensions.

The complete header section is aligned to 8 bytes.

### 2. Keys Table Section

The **Keys Table** provides a mapping of string keys to the respective arrays.
Each entry in the **Keys Table** holds metadata related to one key-value pair.

Each key entry consists of:

- **Key Length (4 bytes)**: The length of the key string (excluding the null terminator).
- **Key Offset (8 bytes)**: Absolute offset, in bytes, to the key string in the **Keys Area**.
- **Data Type (4 bytes)**: Encoded type identifier for the associated data array.
- **Data Length (8 bytes)**: Number of elements in the array.
- **Data Offset (8 bytes)**: Absolute offset where the array begins in the **Data Blocks**.

Keys are stored in lexicographical order to enable binary search for fast access.
The **Keys Table** itself is packed for compactness and aligned to 8-byte boundaries to optimize for random access.

### 3. Data Blocks Section

The **Data Blocks** section holds the arrays themselves, stored sequentially in binary format. Each data block has the following characteristics:

- **Alignment**: Arrays are aligned to 8 bytes to ensure efficient access and compatibility with hardware architectures.
- **Endianness**: All numeric data is stored in little-endian format, which is standard for most modern platforms. Cross-platform compatibility is ensured by converting data to the host's native endianness on read.
- **Compact Storage**: Arrays are written without additional padding, unless required for alignment.

Within each data block:

- The array is stored as raw binary data, with no additional metadata.
- Arrays can represent simple types (e.g., integers, floats) or structured binary blobs, depending on the use case.

#### Layout Example

The following table illustrates the binary layout of a typical kastore file:

```
+------------------+---------------------------------------------+
| Header           | Magic Number, Version, Metadata            |
+------------------+---------------------------------------------+
| Keys Table       | Key Metadata: Offsets, Data Lengths, Types |
+------------------+---------------------------------------------+
| Keys Area        | Null-terminated string keys                |
+------------------+---------------------------------------------+
| Data Blocks      | Binary-encoded array data                  |
+------------------+---------------------------------------------+
```

Data Integrity
--------------

Kastore files are designed to preserve data integrity during access:

1. **Validation on Open**: The magic number, file version, and offsets are validated when a file is opened. Unsupported or corrupted files return a `KAS_ERR_BAD_FILE_FORMAT` error.
2. **Immutable Keys**: Since kastore files are read-only, the keys and associated data cannot be modified after the file is created. This ensures that the binary layout remains consistent and predictable.
3. **Error Codes**: The library provides detailed error codes for handling unexpected issues during access. For example:
    - `KAS_ERR_NO_MEMORY` for insufficient memory during array extraction.
    - `KAS_ERR_KEY_NOT_FOUND` when attempting to retrieve non-existent keys.

Supported Data Types
--------------------

Kastore supports the following common data types, represented by numeric identifiers:

- `KAS_INT8` (0): 8-bit signed integers
- `KAS_UINT8` (1): 8-bit unsigned integers
- `KAS_INT16` (2): 16-bit signed integers
- `KAS_UINT16` (3): 16-bit unsigned integers
- `KAS_INT32` (4): 32-bit signed integers
- `KAS_UINT32` (5): 32-bit unsigned integers
- `KAS_INT64` (6): 64-bit signed integers
- `KAS_UINT64` (7): 64-bit unsigned integers
- `KAS_FLOAT32` (8): 32-bit IEEE-754 floating-point numbers
- `KAS_FLOAT64` (9): 64-bit IEEE-754 floating-point numbers
