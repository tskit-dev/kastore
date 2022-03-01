--------------------
[2.1.1] - 2021-03-01
--------------------

- Minor bug-release and maintenance update.

- Fix assertion triggered when NULL was passed along with KAS_BORROWS_ARRAY.
  (:user:`benjeffery`, :pr:`185`)

- Move VERSION to VERSION.txt to prevent issues on macos.
  (:user:`benjeffery`, :pr:`187`, :issue:`186`)

--------------------
[2.1.0] - 2022-01-25
--------------------

- Add flag ``KAS_BORROWS_ARRAY`` to put. When specified kastore will not copy
  or free the array, which must persist for the life of the store.
  (:user:`benjeffery`, :pr:`181`, :issue:`180`).

- Add flag ``KAS_GET_TAKES_OWNERSHIP`` to open. If specified all ``get`` 
  operations will transfer ownership of the array to the caller. 
  ``kastore`` will not ``free`` the array memory and this is the
  responsibility of the caller.
  (:user:`benjeffery`, :pr:`179`, :issue:`176`)

--------------------
[2.0.1] - 2021-11-26
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


