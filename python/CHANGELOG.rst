--------------------
[0.3.4] - 2025-09-11
--------------------

Maintenance release:

- Require Python >= 3.10, support to 3.13

--------------------
[0.3.3] - 2024-06-25
--------------------

Maintenance release:

- Support for numpy 1.x and 2.x
- Require Python >= 3.9


--------------------
[0.3.2] - 2022-07-12
--------------------

Upgrade build system to build module. Fixes numpy version inconsistency as sdist didn't
specify oldest-supported-numpy. (:user:`benjeffery`, :issue:`190`, :pr:`191`)

--------------------
[0.3.1] - 2020-07-06
--------------------

Fix for packaging bug with numpy version on conda.

--------------------
[0.3.0] - 2020-05-23
--------------------

**New features**

- Support for file-like objects in dump/load and remove
  file seeks. This allows reading from a pipes/FIFOs etc
  (:user:`grahamgower`, :pr:`88`).

- Add ``loads`` and ``dumps`` functions that operate on
  strings (:user:`jeromekelleher`, :pr:`88`)

**Breaking changes**

- The ``filename`` named argument to load/dump has been changed to
  ``file`` to reflect the support for file objects.

- Minimum python version is now 3.6.

--------------------
[0.2.2] - 2018-12-07
--------------------

Fix for packaging bug happening in setup.py bootstrap.

--------------------
[0.2.1] - 2018-12-07
--------------------

Update fixing various packaging and long term API issues.
Export the 0.1.0 C API via the Python Capsule interface.

--------------------
[0.2.0] - 2018-11-20
--------------------

Experimental C API exported via Python Capsule interface.

--------------------
[0.1.0] - 2018-05-06
--------------------

Initial release. Python API only with no public documentation.
