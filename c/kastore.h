#ifndef KASTORE_H
#define KASTORE_H

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

#define KAS_FILE_VERSION_MAJOR 0
#define KAS_FILE_VERSION_MINOR 1

#define KAS_INT8          0
#define KAS_UINT8         1
#define KAS_INT32         2
#define KAS_UINT32        3
#define KAS_INT64         4
#define KAS_UINT64        5
#define KAS_FLOAT32       6
#define KAS_FLOAT64       7
#define KAS_NUM_TYPES     8

#define KAS_READ          0
#define KAS_WRITE         1

#define KAS_HEADER_SIZE             64
#define KAS_ITEM_DESCRIPTOR_SIZE    64
#define KAS_MAGIC                   "\211KAS\r\n\032\n"

typedef struct {
    /* Public attributes denoting the key and array pointers and size.*/
    int type;
    size_t key_len;
    size_t array_len;
    const char *key;
    const void *array;
    /* Private internal fields */
    size_t key_start;
    size_t array_start;
} kaitem_t;

typedef struct {
    int mode;
    int file_version[2];
    size_t num_items;
    kaitem_t *items;
    FILE *file;
    const char *filename;
    size_t file_size;
    char *read_buffer;
} kastore_t;

int kastore_open(kastore_t *self, const char *filename, const char *mode, int flags);
int kastore_close(kastore_t *self);

int kastore_get(kastore_t *self, const char *key, kaitem_t **item, int flags);
int kastore_put(kastore_t *self, const char *key, size_t key_len,
       const void *array, size_t array_len, int type, int flags);

/* Debugging */
void kastore_print_state(kastore_t *self, FILE *out);

#endif
