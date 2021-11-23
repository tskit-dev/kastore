
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>
#include "kastore.h"

/* TskitException is the superclass of all exceptions that can be thrown by
 * tskit. We define it here in the low-level library so that exceptions defined
 * here and in the high-level library can inherit from it.
 */
static PyObject *KastoreException;
static PyObject *KastoreFileFormatError;
static PyObject *KastoreVersionTooOldError;
static PyObject *KastoreVersionTooNewError;

static void
handle_library_error(int err)
{
    switch (err) {
        case KAS_ERR_IO:
            PyErr_SetFromErrno(PyExc_OSError);
            break;
        case KAS_ERR_BAD_FILE_FORMAT:
            PyErr_Format(KastoreFileFormatError, "Bad file format");
            break;
        case KAS_ERR_BAD_TYPE:
            PyErr_Format(KastoreFileFormatError, "Unknown data type");
            break;
        case KAS_ERR_VERSION_TOO_OLD:
            PyErr_SetNone(KastoreVersionTooOldError);
            break;
        case KAS_ERR_VERSION_TOO_NEW:
            PyErr_SetNone(KastoreVersionTooNewError);
            break;
        case KAS_ERR_EOF:
            PyErr_Format(PyExc_EOFError, "Unexpected end of file");
            break;
        default:
            PyErr_Format(
                PyExc_ValueError, "Error occured: %d: %s", err, kas_strerror(err));
    }
}

/*===================================================================
 * Module level code.
 *===================================================================
 */

static int
np_dtype_to_ka_type(int dtype)
{
    int ret = -1;

    switch (dtype) {
        case NPY_INT8:
            ret = KAS_INT8;
            break;
        case NPY_UINT8:
            ret = KAS_UINT8;
            break;
        case NPY_INT16:
            ret = KAS_INT16;
            break;
        case NPY_UINT16:
            ret = KAS_UINT16;
            break;
        case NPY_INT32:
            ret = KAS_INT32;
            break;
        case NPY_UINT32:
            ret = KAS_UINT32;
            break;
        case NPY_INT64:
            ret = KAS_INT64;
            break;
        case NPY_UINT64:
            ret = KAS_UINT64;
            break;
        case NPY_FLOAT32:
            ret = KAS_FLOAT32;
            break;
        case NPY_FLOAT64:
            ret = KAS_FLOAT64;
            break;
    }
    return ret;
}

static int
ka_type_to_np_dtype(int type)
{
    /* We don't do any error checking here because we assume bad types
     * are caught at load time. */
    int ret = NPY_INT8;
    switch (type) {
        case KAS_INT8:
            ret = NPY_INT8;
            break;
        case KAS_UINT8:
            ret = NPY_UINT8;
            break;
        case KAS_INT16:
            ret = NPY_INT16;
            break;
        case KAS_UINT16:
            ret = NPY_UINT16;
            break;
        case KAS_INT32:
            ret = NPY_INT32;
            break;
        case KAS_UINT32:
            ret = NPY_UINT32;
            break;
        case KAS_INT64:
            ret = NPY_INT64;
            break;
        case KAS_UINT64:
            ret = NPY_UINT64;
            break;
        case KAS_FLOAT32:
            ret = NPY_FLOAT32;
            break;
        case KAS_FLOAT64:
            ret = NPY_FLOAT64;
            break;
    }
    return ret;
}

static PyObject *
build_dictionary(kastore_t *store)
{
    PyObject *ret = NULL;
    PyObject *key = NULL;
    PyArrayObject *array = NULL;
    PyObject *data = PyDict_New();
    kaitem_t *item;
    npy_intp dims;
    int dtype, dummy_type, err;
    size_t j, dummy_len;
    void *dummy_array;

    if (data == NULL) {
        goto out;
    }
    for (j = 0; j < store->num_items; j++) {
        item = store->items + j;
        /* When we're opened in the default mode, the item array pointers
         * aren't set until after 'get' is called. So, we have to kludge
         * around this until we have a proper API for accessing all items
         */
        err = kastore_get(
            store, item->key, item->key_len, &dummy_array, &dummy_len, &dummy_type);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
        key = PyUnicode_FromStringAndSize(item->key, item->key_len);
        if (key == NULL) {
            goto out;
        }
        dims = item->array_len;
        dtype = ka_type_to_np_dtype(item->type);
        array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, dtype);
        if (array == NULL) {
            goto out;
        }
        memcpy(
            PyArray_DATA(array), item->array, item->array_len * PyArray_ITEMSIZE(array));
        if (PyDict_SetItem(data, key, (PyObject *) array) != 0) {
            goto out;
        }
        Py_XDECREF(key);
        Py_XDECREF(array);
        key = NULL;
        array = NULL;
    }
    ret = data;
    data = NULL;
out:
    Py_XDECREF(data);
    Py_XDECREF(key);
    Py_XDECREF(array);
    return ret;
}

static int
parse_dictionary(kastore_t *store, PyObject *data)
{
    int ret = -1;
    PyObject *py_key, *py_value;
    PyObject *encoded_key = NULL;
    PyArrayObject *array = NULL;
    Py_ssize_t pos = 0;
    Py_ssize_t key_len;
    npy_intp *shape;
    char *key;
    int err, type;

    while (PyDict_Next(data, &pos, &py_key, &py_value)) {
        if (!PyUnicode_Check(py_key)) {
            PyErr_SetString(PyExc_TypeError, "Keys must be unicode.");
            goto out;
        }
        encoded_key = PyUnicode_AsEncodedString(py_key, "utf-8", "strict");
        if (encoded_key == NULL) {
            /* This error is very difficult/impossible to provoke because
             * PyUnicode_Check makes sure it's unicode, and you can't make
             * invalid unicode strings. */
            goto out;
        }
        if (PyBytes_AsStringAndSize(encoded_key, &key, &key_len) != 0) {
            goto out;
        }
        /* This ensures that only 1D arrays are accepted. */
        array = (PyArrayObject *) PyArray_FromAny(
            py_value, NULL, 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
        if (array == NULL) {
            goto out;
        }
        shape = PyArray_DIMS(array);
        type = np_dtype_to_ka_type(PyArray_DTYPE(array)->type_num);
        if (type < 0) {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype.");
            goto out;
        }
        err = kastore_put(store, key, (size_t) key_len, PyArray_DATA(array),
            (size_t) shape[0], type, 0);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
        Py_DECREF(encoded_key);
        Py_DECREF(array);
        encoded_key = NULL;
        array = NULL;
    }
    ret = 0;
out:
    Py_XDECREF(encoded_key);
    Py_XDECREF(array);
    return ret;
}

static FILE *
make_file(PyObject *fileobj, const char *mode)
{
    FILE *ret = NULL;
    FILE *file = NULL;
    int fileobj_fd, new_fd;

    fileobj_fd = PyObject_AsFileDescriptor(fileobj);
    if (fileobj_fd == -1) {
        goto out;
    }
    new_fd = dup(fileobj_fd);
    if (new_fd == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    file = fdopen(new_fd, mode);
    if (file == NULL) {
        (void) close(new_fd);
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    ret = file;
out:
    return ret;
}

static PyObject *
kastore_load(PyObject *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    kastore_t store;
    PyObject *data = NULL;
    PyObject *fileobj = NULL;
    FILE *file = NULL;
    static char *kwlist[] = { "file", "read_all", NULL };
    int read_all = 0;
    int flags = 0;

    memset(&store, 0, sizeof(store));

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|i:load", kwlist, &fileobj, &read_all)) {
        goto out;
    }
    if (read_all) {
        flags = KAS_READ_ALL;
    }
    file = make_file(fileobj, "rb");
    if (file == NULL) {
        goto out;
    }
    /* Set unbuffered mode to ensure no more bytes are read than requested.
     * Buffered reads could read beyond the end of the current store in a
     * multi-store file or stream. This data would be discarded when we
     * fclose() the file below, such that attempts to load the next store
     * will fail. */
    if (setvbuf(file, NULL, _IONBF, 0) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    err = kastore_openf(&store, file, "r", flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    data = build_dictionary(&store);
    if (data == NULL) {
        goto out;
    }
    err = kastore_close(&store);
    if (err != 0) {
        handle_library_error(err);
    }
    ret = data;
    data = NULL;
out:
    /* In the error case, we ignore errors that occur here. It's OK to call
     * close on a store that has already been closed. */
    kastore_close(&store);
    if (file != NULL) {
        (void) fclose(file);
    }
    Py_XDECREF(data);
    return ret;
}

static PyObject *
kastore_dump(PyObject *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    kastore_t store;
    PyObject *data = NULL;
    PyObject *fileobj = NULL;
    FILE *file = NULL;
    static char *kwlist[] = { "data", "file", NULL };

    memset(&store, 0, sizeof(store));

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O:dump", kwlist, &PyDict_Type, &data, &fileobj)) {
        goto out;
    }

    file = make_file(fileobj, "wb");
    if (file == NULL) {
        goto out;
    }
    err = kastore_openf(&store, file, "w", 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    if (parse_dictionary(&store, data) != 0) {
        goto out;
    }
    err = kastore_close(&store);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    /* In the error case, we ignore errors that occur here. It's OK to call
     * close on a store that has already been closed. */
    kastore_close(&store);
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyMethodDef kastore_methods[] = {
    { "load", (PyCFunction) kastore_load, METH_VARARGS | METH_KEYWORDS,
        "Loads a store from the specified file name." },
    { "dump", (PyCFunction) kastore_dump, METH_VARARGS | METH_KEYWORDS,
        "Writes a store to the specified file name." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef kastoremodule = { PyModuleDef_HEAD_INIT, "_kastore",
    "C interface for kastore.", -1, kastore_methods, NULL, NULL, NULL, NULL };

PyObject *
PyInit__kastore(void)
{
    PyObject *module = PyModule_Create(&kastoremodule);
    kas_version_t version;

    if (module == NULL) {
        return NULL;
    }
    /* Initialise numpy */
    import_array();

    KastoreException = PyErr_NewException("_kastore.KastoreException", NULL, NULL);
    Py_INCREF(KastoreException);
    PyModule_AddObject(module, "KastoreException", KastoreException);

    KastoreFileFormatError
        = PyErr_NewException("_kastore.FileFormatError", KastoreException, NULL);
    Py_INCREF(KastoreFileFormatError);
    PyModule_AddObject(module, "FileFormatError", KastoreFileFormatError);

    KastoreVersionTooOldError
        = PyErr_NewException("_kastore.VersionTooOldError", KastoreException, NULL);
    Py_INCREF(KastoreVersionTooOldError);
    PyModule_AddObject(module, "VersionTooOldError", KastoreVersionTooOldError);

    KastoreVersionTooNewError
        = PyErr_NewException("_kastore.VersionTooNewError", KastoreException, NULL);
    Py_INCREF(KastoreVersionTooNewError);
    PyModule_AddObject(module, "VersionTooNewError", KastoreVersionTooNewError);

    /* Sanity check: we've compiled the kastore code into this module, so there's no
     * way the versions could have changed */
    version = kas_version();
    if (version.major != KAS_VERSION_MAJOR || version.minor != KAS_VERSION_MINOR) {
        PyErr_SetString(PyExc_RuntimeError, "API version mismatch");
        return NULL;
    }
    return module;
}
