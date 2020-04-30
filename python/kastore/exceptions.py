"""
Exception definitions for kastore.
"""
from _kastore import FileFormatError
from _kastore import KastoreException
from _kastore import VersionTooNewError
from _kastore import VersionTooOldError

# Some exceptions are defined in the low-level module. In particular, the
# superclass of all exceptions for kastore is defined here. We define the
# docstrings here to avoid difficulties with compiling C code on
# readthedocs.

KastoreException.__doc__ = "Superclass of all exceptions defined in kastore."
FileFormatError.__doc__ = "An error was detected in the file format."
VersionTooNewError.__doc__ = (
    "File version is too new. Please upgrade your kastore library version "
    "if you wish to read this file."
)
VersionTooOldError.__doc__ = (
    "File version is too old. Please upgrade using the 'kastore upgrade' "
    "command line utility."
)


class StoreClosedError(KastoreException):
    """
    The store has been closed and cannot be accessed.
    """
