import contextlib
import functools
import io
import os.path

from . import _version
from . import store
from .exceptions import *  # noqa

__version__ = _version.kastore_version

_kastore_loaded = True
try:
    import _kastore
except ImportError:  # pragma: no cover
    _kastore_loaded = False
    pass

PY_ENGINE = "python"
C_ENGINE = "c"

DEFAULT_KEY_ENCODING = "utf-8"


def _check_low_level_module():
    if not _kastore_loaded:
        raise RuntimeError("C engine not available")


def _raise_unknown_engine():
    raise ValueError("unknown engine")


def loads(encoded_data, key_encoding=DEFAULT_KEY_ENCODING):
    """
    Loads a store from the specified bytes object.

    :param bytes encoded_data: The encoded kastore data as returned by :func:`.dumps`
        or read from a file written by :func:`.dump`.
    :param str key_encoding: The encoding to use when converting the keys from
        raw bytes.
    :return: A dict-like object mapping the key-array pairs.
    """
    fileobj = io.BytesIO(encoded_data)
    return load(fileobj, key_encoding=key_encoding, read_all=True, engine=PY_ENGINE)


def dumps(data, key_encoding=DEFAULT_KEY_ENCODING):
    """
    Encodes the specified data in kastore form and returns the resulting bytes.

    :param dict data: A dictionary-like string keys to numpy arrays.
    :param str key_encoding: The encoding to use when converting the keys
        to raw bytes.
    :return: The bytes encoding of the specified data in kastore format.
    :rtype: bytes
    """
    fileobj = io.BytesIO()
    dump(data, fileobj, key_encoding=key_encoding, engine=PY_ENGINE)
    return fileobj.getvalue()


@contextlib.contextmanager
def _open_file(fileobj, mode):
    """
    Abstracts the details of opening file-like objects and freeing the
    resources used.
    """
    # First, see if we can interpret the argument as a pathlike object.
    path = None
    try:
        path = os.fspath(fileobj)
    except TypeError:
        pass
    if path is not None:
        with open(path, mode) as f:
            yield f
    else:
        # Now we try to open fileobj. If it's not a pathlike object, it could be
        # an integer fd. In this case we must make sure that close is **not**
        # called on the fd. It's also important to set buffering to zero for the
        # Python engine.
        fd = None
        try:
            fd = int(fileobj)
        except TypeError:
            pass
        if fd is not None:
            with open(fileobj, mode, closefd=False, buffering=0) as f:
                yield f
        else:
            # Finally, this could a fileobj-like object in itself. In this case
            # we return it as-is and don't close it. We retain TypeError semantics
            # from earlier versions.
            if not hasattr(fileobj, "write"):
                raise TypeError("fileobj object must have a write method")
            yield fileobj


def load(file, read_all=False, key_encoding=DEFAULT_KEY_ENCODING, engine=PY_ENGINE):
    """
    Loads a store from the specified file.

    :param str file: The path of the file to load, or a file-like object
        with a ``read()`` method.
    :param bool read_all: If True, read the entire file into memory. This
        optimisation is useful when all the data will be needed, saving some
        malloc and fread overhead.
    :param str key_encoding: The encoding to use when converting the keys from
        raw bytes.
    :param str engine: The underlying implementation to use.
    :return: A dict-like object mapping the key-array pairs.
    """
    if engine == PY_ENGINE:
        # The Python engine returns an object which needs to keep its own
        # copy of the file object, so must handle the semantics of opening
        # itself.
        ret = store.load(file, key_encoding=key_encoding, read_all=read_all)

    elif engine == C_ENGINE:
        _check_low_level_module()
        with _open_file(file, "rb") as f:
            ret = _kastore.load(f, read_all=read_all)
    else:
        _raise_unknown_engine()
    return ret


def dump(data, file, key_encoding=DEFAULT_KEY_ENCODING, engine=PY_ENGINE):
    """
    Dumps a store to the specified file.

    :param dict data: A dictionary-like string keys to numpy arrays.
    :param str file: The path of the file to write the store to, or a
        file-like object with a ``write()`` method.
    :param str key_encoding: The encoding to use when converting the keys
        to raw bytes.
    :param str engine: The underlying implementation to use.
    """
    if engine == PY_ENGINE:
        dump_func = functools.partial(store.dump, key_encoding=key_encoding)
    elif engine == C_ENGINE:
        _check_low_level_module()
        dump_func = _kastore.dump
    else:
        _raise_unknown_engine()

    with _open_file(file, "wb") as f:
        dump_func(data, f)


def get_include():
    """
    Returns the directory path where include files for the kastore C API are
    to be found.
    """
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_path, "include")
