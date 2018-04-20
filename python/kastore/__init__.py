from __future__ import print_function
from __future__ import division

from . import store
import _kastore

PY_ENGINE = "python"
C_ENGINE = "c"


def _raise_unknown_engine():
    raise ValueError("unknown engine")


def load(filename, key_encoding="utf-8", engine=PY_ENGINE):
    if engine == PY_ENGINE:
        return store.load(filename, key_encoding)
    elif engine == C_ENGINE:
        return _kastore.load(filename)
    else:
        _raise_unknown_engine()


def dump(data, filename, key_encoding="utf-8", engine=PY_ENGINE):
    if engine == PY_ENGINE:
        store.dump(data, filename, key_encoding)
    elif engine == C_ENGINE:
        _kastore.dump(data, filename)
    else:
        _raise_unknown_engine()


def main():
    print("Main entry point for kastore command")
