from __future__ import print_function
from __future__ import division

__version__ = "0.1.0"

from . import store
from . exceptions import FileFormatError
from . exceptions import VersionTooOldError
from . exceptions import VersionTooNewError
import _kastore

PY_ENGINE = "python"
C_ENGINE = "c"


def _raise_unknown_engine():
    raise ValueError("unknown engine")


def load(filename, use_mmap=True, key_encoding="utf-8", engine=PY_ENGINE):
    if engine == PY_ENGINE:
        return store.load(filename, use_mmap=use_mmap, key_encoding=key_encoding)
    elif engine == C_ENGINE:
        try:
            return _kastore.load(filename, use_mmap=use_mmap)
        except _kastore.FileFormatError as e:
            # Note in Python 3 we should use "raise X from e" to designate
            # that the low-level exception is the cause of the high-level
            # exception. We can't do that in Python 2 though, and it's not
            # worth having separate code paths. Same for all the other
            # exceptions we're chaining here.
            raise FileFormatError(str(e))
        except _kastore.VersionTooOldError as e:
            raise VersionTooOldError()
        except _kastore.VersionTooNewError as e:
            raise VersionTooNewError()

    else:
        _raise_unknown_engine()


def dump(data, filename, key_encoding="utf-8", engine=PY_ENGINE):
    if engine == PY_ENGINE:
        store.dump(data, filename, key_encoding)
    elif engine == C_ENGINE:
        _kastore.dump(data, filename)
    else:
        _raise_unknown_engine()
