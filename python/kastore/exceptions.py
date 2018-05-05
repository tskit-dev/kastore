"""
Exception definitions for kastore.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


class KastoreException(Exception):
    """
    Parent class of all Kastore specific exceptions.
    """


class FileFormatError(KastoreException):
    """
    The provided file was not in the expected format.
    """


class StoreClosedError(KastoreException):
    """
    The store has been closed and cannot be accessed.
    """


class VersionTooOldError(KastoreException):
    """
    The provided file is too old to be read by the current version
    of kastore.
    """
    def __init__(self):
        super(VersionTooOldError, self).__init__(
            "File version is too old. Please upgrade using the 'kastore upgrade' "
            "command line utility")


class VersionTooNewError(KastoreException):
    """
    The provided file is too old to be read by the current version
    of kastore.
    """
    def __init__(self):
        super(VersionTooNewError, self).__init__(
            "File version is too new. Please upgrade your kastore library version "
            "if you wish to read this file.")
