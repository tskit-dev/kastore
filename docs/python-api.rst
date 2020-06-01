.. _sec_python_api:

========================
Python API Documentation
========================

Kastore provides a simple Python interface to loading and storing
key-array mappings in :ref:`kastore format <sec_format>`.

.. autofunction:: kastore.load
.. autofunction:: kastore.loads
.. autofunction:: kastore.dump
.. autofunction:: kastore.dumps
.. autofunction:: kastore.get_include

**********
Exceptions
**********

.. autoexception:: kastore.KastoreException
.. autoexception:: kastore.FileFormatError
.. autoexception:: kastore.VersionTooNewError
.. autoexception:: kastore.VersionTooOldError

*******
Example
*******

Here is a simple example of using kastore to save some numpy arrays
to a file and load them again.

.. code-block:: python

    import kastore
    import numpy as np

    data = {"one": np.arange(5, dtype=np.int8), "two": np.arange(5, dtype=np.uint64)}
    kastore.dump(data, "tmp.kas")

    kas = kastore.load("tmp.kas")
    print(list(kas.items()))


Running this code chunk gives us::

    [('one', array([0, 1, 2, 3, 4], dtype=int8)), ('two', array([0, 1, 2, 3, 4], dtype=uint64))]


We can also get a useful summmary of a kastore file using the command line interface::

    $ python3 -m kastore ls -lH tmp.kas
    int8   5  5B one
    uint64 5 40B two

The output here shows us that the array corresponding to key ``one``
has type ``int8``, 5 elements and consumes 5 bytes of space. The array for key ``two``
also have 5 elements but has type ``uint64`` and therefore consumes 40 bytes of space.

Please see the output of ``python3 -m kastore --help`` for more help on this
command line interface.

The python module gives a *read-only* view of the kastore file,
so to add more data to an existing store, we need to load it,
convert it to a dict (which is efficient, as the underlying arrays won't be copied),
and dump it back out again.
For instance, here's how we might add a new key to the previous example:

.. code-block:: python

   kas_dict = dict(kas)
   print(kas_dict)
   # {'one': array([0, 1, 2, 3, 4], dtype=int8), 'two': array([0, 1, 2, 3, 4], dtype=uint64)}

   kas_dict["three"] = np.array([0.5772, 2.7818, 3.1415])
   kastore.dump(kas_dict, "tmp2.kas")

After this, we get::

   # python3 -m kastore ls -lH tmp2.kas
   int8    5  5B one
   float64 3 24B three
   uint64  5 40B two

indicating that the key "three" has three float64 entries.
