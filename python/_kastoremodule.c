
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "kastore.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define MODULE_DOC \
"C interface for kastore."

/* static PyObject *TsinfLibraryError; */

static void
handle_library_error(int err)
{
    if (err == KAS_ERR_IO) {
        PyErr_SetFromErrno(PyExc_ValueError);
    } else {
        PyErr_Format(PyExc_ValueError, "Error occured: %d", err);
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
    int dtype;
    size_t j;

    if (data == NULL) {
        goto out;
    }
    for (j = 0; j < store->num_items; j++) {
        item = store->items + j;
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
        memcpy(PyArray_DATA(array), item->array,
                item->array_len * PyArray_ITEMSIZE(array));
        if (PyDict_SetItem(data, key, (PyObject *) array) != 0) {
            goto out;
        }
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
            goto out;
        }
        if (PyBytes_AsStringAndSize(encoded_key, &key, &key_len) != 0) {
            goto out;
        }
        /* This ensures that only 1D arrays are accepted. */
        array = (PyArrayObject *) PyArray_FromAny(py_value, NULL, 1, 1,
                NPY_ARRAY_IN_ARRAY, NULL);
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

static PyObject *
kastore_load(PyObject *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    char *filename;
    kastore_t store;
    PyObject *data = NULL;
    static char *kwlist[] = {"filename", NULL};

    memset(&store, 0, sizeof(store));

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:load", kwlist,
                &filename)) {
        goto out;
    }
    err = kastore_open(&store, filename, "r", 0);
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
     * close on store that has already been closed. */
    kastore_close(&store);
    Py_XDECREF(data);
    return ret;
}

static PyObject *
kastore_dump(PyObject *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    char *filename;
    kastore_t store;
    PyObject *data = NULL;
    static char *kwlist[] = {"data", "filename", NULL};

    memset(&store, 0, sizeof(store));

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!s:dump", kwlist,
                &PyDict_Type, &data, &filename)) {
        goto out;
    }
    err = kastore_open(&store, filename, "w", 0);
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
     * close on store that has already been closed. */
    kastore_close(&store);
    return ret;
}

static PyMethodDef kastore_methods[] = {
    {"load", (PyCFunction) kastore_load, METH_VARARGS|METH_KEYWORDS,
            "Loads a store from the specified file name." },
    {"dump", (PyCFunction) kastore_dump, METH_VARARGS|METH_KEYWORDS,
            "Writes a store to the specified file name." },
    {NULL}        /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef kastoremodule = {
    PyModuleDef_HEAD_INIT,
    "_kastore",   /* name of module */
    MODULE_DOC, /* module documentation, may be NULL */
    -1,
    kastore_methods,
    NULL, NULL, NULL, NULL
};

#define INITERROR return NULL

PyObject *
PyInit__kastore(void)

#else
#define INITERROR return

void
init_kastore(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&kastoremodule);
#else
    PyObject *module = Py_InitModule3("_kastore", kastore_methods, MODULE_DOC);
#endif
    if (module == NULL) {
        INITERROR;
    }
    /* Initialise numpy */
    import_array();

    /* TsinfLibraryError = PyErr_NewException("_kastore.LibraryError", NULL, NULL); */
    /* Py_INCREF(TsinfLibraryError); */
    /* PyModule_AddObject(module, "LibraryError", TsinfLibraryError); */

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}