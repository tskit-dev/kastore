.. _sec_c_api:

===================
C API Documentation
===================

This is the C API documentation for kastore.

.. _sec_c_api_example:

***************
Example program
***************

.. literalinclude:: ../c/example.c
    :language: c

******************
General principles
******************

--------------
Error handling
--------------

Functions return 0 to indicate success or an
:ref:`error code <sec_c_api_error_codes>` to indicate a failure condition.
Thus, the return value of all functions must be checked to ensure safety.

-------------
Array lengths
-------------

The length of arrays is specified in terms of the number of elements not bytes.

*********
Top level
*********

.. doxygenstruct:: kastore_t

.. doxygenfunction:: kastore_open
.. doxygenfunction:: kastore_openf
.. doxygenfunction:: kastore_close

.. doxygenfunction:: kas_strerror

.. _sec_c_api_get:


******************
Contains functions
******************

Contains functions provide a way to determine if a given key is in the store.

.. doxygenfunction:: kastore_contains
.. doxygenfunction:: kastore_containss

*************
Get functions
*************

Get functions provide the interface for querying a store. The most general interface
is :c:func:`kastore_get`, but it is usually more convenient to use one of the
:ref:`typed get functions <sec_c_api_typed_get>`.

.. doxygenfunction:: kastore_get
.. doxygenfunction:: kastore_gets

.. _sec_c_api_typed_get:

----------
Typed gets
----------

The functions listed here provide a convenient short-cut for accessing arrays
where the key is a standard NULL terminated C string and the type of the
array is known in advance.

.. doxygengroup:: TYPED_GETS_GROUP
        :content-only:

.. _sec_c_api_put:

*************
Put functions
*************

Put functions provide the interface for inserting data into store. The most
general interface is :c:func:`kastore_put` which allows keys to be arbitrary
bytes, but it is usually more convenient to use one of the :ref:`typed put
functions <sec_c_api_typed_put>`.

.. doxygenfunction:: kastore_put
.. doxygenfunction:: kastore_puts

.. _sec_c_api_typed_put:

----------
Typed puts
----------

The functions listed here provide a convenient short-cut for inserting
key-array pairs where the key is a standard NULL terminated C string and the
type of the array is known in advance.

.. doxygengroup:: TYPED_PUTS_GROUP
        :content-only:


.. _sec_c_api_oput:

*****************
Own-put functions
*****************

The 'own-put' functions are almost identical to the standard 'put' functions,
but transfer ownership of the array buffer from the caller to the store. This
is useful, for example, when client code wishes to write a large array to the
store and wants of avoid the overhead of keeping a separate copy of this buffer
in the store. By calling :c:func:`kastore_oput`, the user can put the key-array
pair into the store and transfer responsibility for freeing the malloced
array buffer to the store. See the :ref:`sec_c_api_example` for an illustration.

.. doxygenfunction:: kastore_oput
.. doxygenfunction:: kastore_oputs

.. _sec_c_api_typed_oput:

-----------
Typed oputs
-----------

.. doxygengroup:: TYPED_OPUTS_GROUP
        :content-only:


*********
Constants
*********

.. _sec_c_api_error_codes:

------
Errors
------

.. doxygengroup:: ERROR_GROUP
        :content-only:

-----
Types
-----

.. doxygengroup:: TYPE_GROUP
        :content-only:

-------------------
Version information
-------------------

.. doxygengroup:: API_VERSION_GROUP
        :content-only:

.. doxygengroup:: FILE_VERSION_GROUP
        :content-only:

***********************
Miscellaneous functions
***********************

.. doxygenstruct:: kas_version_t
    :members:

.. doxygenfunction:: kas_version

