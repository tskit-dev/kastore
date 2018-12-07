from __future__ import print_function
from __future__ import division

import os.path

from . import store
from . exceptions import FileFormatError
from . exceptions import VersionTooOldError
from . exceptions import VersionTooNewError
from . import _version
__version__ = _version.kastore_version

_kastore_loaded = True
try:
    import _kastore
except ImportError:  # pragma: no cover
    _kastore_loaded = False
    pass

PY_ENGINE = "python"
C_ENGINE = "c"


def _check_low_level_module():
    if not _kastore_loaded:
        raise RuntimeError("C engine not available")


def _raise_unknown_engine():
    raise ValueError("unknown engine")


def load(filename, read_all=False, key_encoding="utf-8", engine=PY_ENGINE):
    """
    Loads a store from the specified file.

    :param bool read_all: If True, read the entire file into memory. This
        optimisation is useful when all the data will be needed, saving some
        malloc and fread overhead.
    :param str filename: The path of the file to load.
    :param str key_encoding: The encoding to use when converting the keys from
        raw bytes.
    :param str engine: The underlying implementation to use.
    :return: A dict-like object mapping the key-array pairs.
    """
    if engine == PY_ENGINE:
        return store.load(filename, read_all=read_all, key_encoding=key_encoding)
    elif engine == C_ENGINE:
        _check_low_level_module()
        try:
            return _kastore.load(filename, read_all=read_all)
        except _kastore.FileFormatError as e:
            # Note in Python 3 we should use "raise X from e" to designate
            # that the low-level exception is the cause of the high-level
            # exception. We can't do that in Python 2 though, and it's not
            # worth having separate code paths. Same for all the other
            # exceptions we're chaining here.
            raise FileFormatError(str(e))
        except _kastore.VersionTooOldError:
            raise VersionTooOldError()
        except _kastore.VersionTooNewError:
            raise VersionTooNewError()

    else:
        _raise_unknown_engine()


def dump(data, filename, key_encoding="utf-8", engine=PY_ENGINE):
    """
    Dumps a store to the specified file.

    :param str filename: The path of the file to write the store to.
    :param str key_encoding: The encoding to use when converting the keys
        to raw bytes.
    :param str engine: The underlying implementation to use.
    """
    if engine == PY_ENGINE:
        store.dump(data, filename, key_encoding)
    elif engine == C_ENGINE:
        _check_low_level_module()
        _kastore.dump(data, filename)
    else:
        _raise_unknown_engine()


def get_include():
    """
    Returns the directory path where include files for the kastore C API are
    to be found.
    """
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_path, "include")
