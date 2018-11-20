#ifndef KASTORE_H
#define KASTORE_H

#ifdef __GNUC__
    #define KAS_WARN_UNUSED __attribute__ ((warn_unused_result))
    #define KAS_UNUSED(x) KAS_UNUSED_ ## x __attribute__((__unused__))
#else
    #define KAS_WARN_UNUSED
    #define KAS_UNUSED(x) KAS_UNUSED_ ## x
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define KAS_ERR_GENERIC                               -1
#define KAS_ERR_IO                                    -2
#define KAS_ERR_BAD_MODE                              -3
#define KAS_ERR_NO_MEMORY                             -4
#define KAS_ERR_BAD_FILE_FORMAT                       -5
#define KAS_ERR_VERSION_TOO_OLD                       -6
#define KAS_ERR_VERSION_TOO_NEW                       -7
#define KAS_ERR_BAD_TYPE                              -8
#define KAS_ERR_EMPTY_KEY                             -9
#define KAS_ERR_DUPLICATE_KEY                         -10
#define KAS_ERR_KEY_NOT_FOUND                         -11
#define KAS_ERR_ILLEGAL_OPERATION                     -12
#define KAS_ERR_TYPE_MISMATCH                         -13

/* Flags for open */
#define KAS_NO_MMAP             1

#define KAS_FILE_VERSION_MAJOR  1
#define KAS_FILE_VERSION_MINOR  0

#define KAS_INT8                0
#define KAS_UINT8               1
#define KAS_INT16               2
#define KAS_UINT16              3
#define KAS_INT32               4
#define KAS_UINT32              5
#define KAS_INT64               6
#define KAS_UINT64              7
#define KAS_FLOAT32             8
#define KAS_FLOAT64             9
#define KAS_NUM_TYPES           10

#define KAS_READ                1
#define KAS_WRITE               2

#define KAS_HEADER_SIZE             64
#define KAS_ITEM_DESCRIPTOR_SIZE    64
#define KAS_MAGIC                   "\211KAS\r\n\032\n"
#define KAS_ARRAY_ALIGN             8

typedef struct {
    int type;
    size_t key_len;
    size_t array_len;
    char *key;
    void *array;
    size_t key_start;
    size_t array_start;
} kaitem_t;

typedef struct {
    int flags;
    int mode;
    int file_version[2];
    size_t num_items;
    kaitem_t *items;
    FILE *file;
    const char *filename;
    size_t file_size;
    char *read_buffer;
} kastore_t;


#define KAS_PROTO_OPEN (kastore_t *self, const char *filename, const char *mode, int flags)
#define KAS_PROTO_CLOSE (kastore_t *self)

#define KAS_PROTO_GET (kastore_t *self, const char *key, size_t key_len, \
        void **array, size_t *array_len, int *type)
#define KAS_PROTO_GETS \
    (kastore_t *self, const char *key, void **array, size_t *array_len, int *type)
#define KAS_PROTO_GETS_INT8 \
    (kastore_t *self, const char *key, int8_t **array, size_t *array_len)
#define KAS_PROTO_GETS_UINT8 \
    (kastore_t *self, const char *key, uint8_t **array, size_t *array_len)
#define KAS_PROTO_GETS_INT16 \
    (kastore_t *self, const char *key, int16_t **array, size_t *array_len)
#define KAS_PROTO_GETS_UINT16 \
    (kastore_t *self, const char *key, uint16_t **array, size_t *array_len)
#define KAS_PROTO_GETS_INT32 \
    (kastore_t *self, const char *key, int32_t **array, size_t *array_len)
#define KAS_PROTO_GETS_UINT32 \
    (kastore_t *self, const char *key, uint32_t **array, size_t *array_len)
#define KAS_PROTO_GETS_INT64 \
    (kastore_t *self, const char *key, int64_t **array, size_t *array_len)
#define KAS_PROTO_GETS_UINT64 \
    (kastore_t *self, const char *key, uint64_t **array, size_t *array_len)
#define KAS_PROTO_GETS_FLOAT32 \
    (kastore_t *self, const char *key, float **array, size_t *array_len)
#define KAS_PROTO_GETS_FLOAT64 \
    (kastore_t *self, const char *key, double **array, size_t *array_len)

#define KAS_PROTO_PUT (kastore_t *self, const char *key, size_t key_len, \
       const void *array, size_t array_len, int type, int flags)
#define KAS_PROTO_PUTS (kastore_t *self, const char *key, \
       const void *array, size_t array_len, int type, int flags)
#define KAS_PROTO_PUTS_INT8 (kastore_t *self, const char *key, const int8_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_UINT8 (kastore_t *self, const char *key, const uint8_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_INT16 (kastore_t *self, const char *key, const int16_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_UINT16 (kastore_t *self, const char *key, const uint16_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_INT32 (kastore_t *self, const char *key, const int32_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_UINT32 (kastore_t *self, const char *key, const uint32_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_INT64 (kastore_t *self, const char *key, const int64_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_UINT64 (kastore_t *self, const char *key, const uint64_t *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_FLOAT32 (kastore_t *self, const char *key, const float *array, \
        size_t array_len, int flags)
#define KAS_PROTO_PUTS_FLOAT64 (kastore_t *self, const char *key, const double *array, \
        size_t array_len, int flags)

#define KAS_PROTO_PRINT_STATE (kastore_t *self, FILE *out)
#define KAS_PROTO_STRERROR (int err)

#define KAS_INDEX_OPEN              0
#define KAS_INDEX_CLOSE             1
#define KAS_INDEX_GET               2
#define KAS_INDEX_GETS              3
#define KAS_INDEX_GETS_INT8         4
#define KAS_INDEX_GETS_UINT8        5
#define KAS_INDEX_GETS_INT16        6
#define KAS_INDEX_GETS_UINT16       7
#define KAS_INDEX_GETS_INT32        8
#define KAS_INDEX_GETS_UINT32       9
#define KAS_INDEX_GETS_INT64        10
#define KAS_INDEX_GETS_UINT64       11
#define KAS_INDEX_GETS_FLOAT32      12
#define KAS_INDEX_GETS_FLOAT64      13
#define KAS_INDEX_PUT               14
#define KAS_INDEX_PUTS              15
#define KAS_INDEX_PUTS_INT8         16
#define KAS_INDEX_PUTS_UINT8        17
#define KAS_INDEX_PUTS_INT16        18
#define KAS_INDEX_PUTS_UINT16       19
#define KAS_INDEX_PUTS_INT32        20
#define KAS_INDEX_PUTS_UINT32       21
#define KAS_INDEX_PUTS_INT64        22
#define KAS_INDEX_PUTS_UINT64       23
#define KAS_INDEX_PUTS_FLOAT32      24
#define KAS_INDEX_PUTS_FLOAT64      25
#define KAS_INDEX_PRINT_STATE       26
#define KAS_INDEX_STRERROR          27

/* Total number of exported functions */
#define KAS_DYNAMIC_API_NUM 28

/* We need to pass around arrays of generic function pointers. Because
 * C99 does not allow us to cast function pointers to void *, it's useful
 * to have this shorthand. */
typedef void (*kas_funcptr)(void);

#ifdef KAS_DYNAMIC_API
extern kas_funcptr *kas_dynamic_api;

#define kastore_open (*(int (*)KAS_PROTO_OPEN) kas_dynamic_api[KAS_INDEX_OPEN])
#define kastore_close (*(int (*)KAS_PROTO_CLOSE) kas_dynamic_api[KAS_INDEX_CLOSE])

#define kastore_get (*(int (*)KAS_PROTO_GET) kas_dynamic_api[KAS_INDEX_GET])
#define kastore_gets (*(int (*)KAS_PROTO_GETS) kas_dynamic_api[KAS_INDEX_GETS])
#define kastore_gets_int8 (*(int (*)KAS_PROTO_GETS_INT8) \
        kas_dynamic_api[KAS_INDEX_GETS_INT8])
#define kastore_gets_uint8 (*(int (*)KAS_PROTO_GETS_UINT8) \
        kas_dynamic_api[KAS_INDEX_GETS_UINT8])
#define kastore_gets_int16 (*(int (*)KAS_PROTO_GETS_INT16) \
        kas_dynamic_api[KAS_INDEX_GETS_INT16])
#define kastore_gets_uint16 (*(int (*)KAS_PROTO_GETS_UINT16) \
        kas_dynamic_api[KAS_INDEX_GETS_UINT16])
#define kastore_gets_int32 (*(int (*)KAS_PROTO_GETS_INT32) \
        kas_dynamic_api[KAS_INDEX_GETS_INT32])
#define kastore_gets_uint32 (*(int (*)KAS_PROTO_GETS_UINT32) \
        kas_dynamic_api[KAS_INDEX_GETS_UINT32])
#define kastore_gets_int64 (*(int (*)KAS_PROTO_GETS_INT64) \
        kas_dynamic_api[KAS_INDEX_GETS_INT64])
#define kastore_gets_uint64 (*(int (*)KAS_PROTO_GETS_UINT64) \
        kas_dynamic_api[KAS_INDEX_GETS_UINT64])
#define kastore_gets_float32 (*(int (*)KAS_PROTO_GETS_FLOAT32) \
        kas_dynamic_api[KAS_INDEX_GETS_FLOAT32])
#define kastore_gets_float64 (*(int (*)KAS_PROTO_GETS_FLOAT64) \
        kas_dynamic_api[KAS_INDEX_GETS_FLOAT64])

#define kastore_put (*(int (*)KAS_PROTO_PUT) kas_dynamic_api[KAS_INDEX_PUT])
#define kastore_puts (*(int (*)KAS_PROTO_PUTS) kas_dynamic_api[KAS_INDEX_PUTS])
#define kastore_puts_int8 (*(int (*)KAS_PROTO_PUTS_INT8) \
        kas_dynamic_api[KAS_INDEX_PUTS_INT8])
#define kastore_puts_uint8 (*(int (*)KAS_PROTO_PUTS_UINT8) \
        kas_dynamic_api[KAS_INDEX_PUTS_UINT8])
#define kastore_puts_int16 (*(int (*)KAS_PROTO_PUTS_INT16) \
        kas_dynamic_api[KAS_INDEX_PUTS_INT16])
#define kastore_puts_uint16 (*(int (*)KAS_PROTO_PUTS_UINT16) \
        kas_dynamic_api[KAS_INDEX_PUTS_UINT16])
#define kastore_puts_int32 (*(int (*)KAS_PROTO_PUTS_INT32) \
        kas_dynamic_api[KAS_INDEX_PUTS_INT32])
#define kastore_puts_uint32 (*(int (*)KAS_PROTO_PUTS_UINT32) \
        kas_dynamic_api[KAS_INDEX_PUTS_UINT32])
#define kastore_puts_int64 (*(int (*)KAS_PROTO_PUTS_INT64) \
        kas_dynamic_api[KAS_INDEX_PUTS_INT64])
#define kastore_puts_uint64 (*(int (*)KAS_PROTO_PUTS_UINT64) \
        kas_dynamic_api[KAS_INDEX_PUTS_UINT64])
#define kastore_puts_float32 (*(int (*)KAS_PROTO_PUTS_FLOAT32) \
        kas_dynamic_api[KAS_INDEX_PUTS_FLOAT32])
#define kastore_puts_float64 (*(int (*)KAS_PROTO_PUTS_FLOAT64) \
        kas_dynamic_api[KAS_INDEX_PUTS_FLOAT64])

#define kastore_print_state (*(void (*)KAS_PROTO_PRINT_STATE) \
        kas_dynamic_api[KAS_INDEX_PRINT_STATE])
#define kas_strerror (*(const char * (*)KAS_PROTO_STRERROR) \
        kas_dynamic_api[KAS_INDEX_STRERROR])

#else

int kastore_open KAS_PROTO_OPEN;
int kastore_close KAS_PROTO_CLOSE;

int kastore_get KAS_PROTO_GET;
int kastore_gets KAS_PROTO_GETS;
int kastore_gets_int8 KAS_PROTO_GETS_INT8;
int kastore_gets_uint8 KAS_PROTO_GETS_UINT8;
int kastore_gets_int16 KAS_PROTO_GETS_INT16;
int kastore_gets_uint16 KAS_PROTO_GETS_UINT16;
int kastore_gets_int32 KAS_PROTO_GETS_INT32;
int kastore_gets_uint32 KAS_PROTO_GETS_UINT32;
int kastore_gets_int64 KAS_PROTO_GETS_INT64;
int kastore_gets_uint64 KAS_PROTO_GETS_UINT64;
int kastore_gets_float32 KAS_PROTO_GETS_FLOAT32;
int kastore_gets_float64 KAS_PROTO_GETS_FLOAT64;

int kastore_put KAS_PROTO_PUT;
int kastore_puts KAS_PROTO_PUTS;
int kastore_puts_int8 KAS_PROTO_PUTS_INT8;
int kastore_puts_uint8 KAS_PROTO_PUTS_UINT8;
int kastore_puts_int16 KAS_PROTO_PUTS_INT16;
int kastore_puts_uint16 KAS_PROTO_PUTS_UINT16;
int kastore_puts_int32 KAS_PROTO_PUTS_INT32;
int kastore_puts_uint32 KAS_PROTO_PUTS_UINT32;
int kastore_puts_int64 KAS_PROTO_PUTS_INT64;
int kastore_puts_uint64 KAS_PROTO_PUTS_UINT64;
int kastore_puts_float32 KAS_PROTO_PUTS_FLOAT32;
int kastore_puts_float64 KAS_PROTO_PUTS_FLOAT64;

void kastore_print_state KAS_PROTO_PRINT_STATE;
const char *kas_strerror KAS_PROTO_STRERROR;

kas_funcptr* kas_dynamic_api_init(void);

#endif

#define kas_safe_free(pointer) \
do {\
    if (pointer != NULL) {\
        free(pointer);\
        pointer = NULL;\
    }\
} while (0)

#endif
