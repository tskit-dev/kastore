--------------------
[2.0.1] - 2021-11-xx
--------------------

- Minor bug-release and maintenance update.

**Bug fixes**

- Fix an overflow in an internal flags value on platforms with
  16 bit int types (:user:`jeromekelleher`, :issue:`153`, :pr:`153`).

- Fix a bug in which error conditions were not reported in append
  mode if an error occured when closing the temporary file used
  to read the data. (:user:`jeromekelleher`, :issue:`160`, :pr:`164`).

--------------------
[2.0.0] - 2020-05-23
--------------------

- Major file version bumped because new fields were added to the kastore_t
  struct, leading to potential ABI breakage. No API breakage should occur.

**New features**

- Add kastore_openf function to support FILE objects, and remove
  file seeks. This allows reading from a pipes/FIFOs, and allows
  multiple stores to read from the same stream
  (:user:`grahamgower`, :pr:`88`).

--------------------
[1.1.0] - 2019-03-19
--------------------

- Add `contains` function
- Add `oput` variants that transfer ownership of buffer.
- Various documentation updates.

--------------------
[1.0.1] - 2019-01-24
--------------------

Add support for using kastore as a meson subproject.

--------------------
[1.0.0] - 2019-01-22
--------------------

Remove the dynamic C API option and add support for C++.

--------------------
[0.1.0] - 2018-12-07
--------------------

Initial release of the documented C API.


