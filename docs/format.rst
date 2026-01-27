.. _sec_format:

===========
File format
===========

Binary File Structure
---------------------

The file format layout is as follows.
+-----------------------------------+
| Header (64 bytes)                 |
+-----------------------------------+
| Item descriptors (n * 64 bytes)   |
+-----------------------------------+
| Keys packed densely.              |
+-----------------------------------+
| Arrays packed densely starting on |
| 8 byte boundaries.                |
+-----------------------------------+
   
*********
1. Header
*********

The **Header** occupies a fixed length of **64 bytes** at the beginning of the file.

- **Magic Number (8 bytes)**: Format Signature, derived from the strategy used by HDF5 and PNG.
- **File Version (4 bytes)**: The major and minor version numbers, ensuring forward and backward compatibility.
    - **Major Version (2 bytes)**: Incremented when breaking changes are introduced.
    - **Minor Version (2 bytes)**: Incremented when non-breaking, backward-compatible changes are made.
- **Num Items (4 bytes)**: Number of key-value pairs stored in this file.
- **File Size (8 bytes)**: Size of the file.
- **Reserved (40 bytes)**: Reserved for future extensions.

Format Signature description::

    (decimal)              137  75  65  83  13  10  26  10
    (hexadecimal)           89  4b  41  53  0d  0a  1a  0a
    (ASCII C notation)    \211   K   A   S  \r  \n \032 \n


********************
2. Item descriptors
********************

Block of **Num Items** * **64 bytes**.
The **Item descriptors** provides a mapping for keys and arrays.
Each item descriptor is a block of 64 bytes, which stores:

- The numeric indentifier for the type of the array (between 0 and 9, see Supported Data Types section)
- The start offset of the key
- The length of the key
- The start offset of the array
- The length of the array

File offsets are stored as 8 bytes unsigned little endian integers.

.. table:: 
  :width: 100%
  :align: center

  +------+------+------+------+------+------+------+------+
  | byte | byte | byte | byte | byte | byte | byte | byte |
  +======+======+======+======+======+======+======+======+
  | Type | Reserved                                       |
  +------+------------------------------------------------+
  |        Key start                                      |
  +-------------------------------------------------------+
  |        Key length                                     |
  +-------------------------------------------------------+
  |        Array start                                    |
  +-------------------------------------------------------+
  |        Array length                                   |
  +-------------------------------------------------------+
  |  Reserved (bytes 40:64)  [...]                        |
  +-------------------------------------------------------+


*********
3. Keys
*********

Packed keys encoded in UTF-8.
Keys are stored in sorted order.

*********
4. Arrays
*********

Arrays are packed densely and are 8 byte aligned.


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
