#ifndef KASTORE_H
#define KASTORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define KAS_ERR_GENERIC                               -1
#define KAS_ERR_IO                                    -2
#define KAS_ERR_BAD_MODE                              -3
#define KAS_ERR_NO_MEMORY                             -4

#define KAS_VERSION_MAJOR 0
#define KAS_VERSION_MINOR 1

#define KAS_INT8          0
#define KAS_UINT8         1
#define KAS_INT32         2
#define KAS_UINT32        3
#define KAS_INT64         4
#define KAS_UINT64        5
#define KAS_FLOAT32       6
#define KAS_FLOAT64       7

#define KAS_READ          0
#define KAS_WRITE         1


typedef struct {
    /* Public attributes denoting the key and array pointers and size.*/
    int type;
    size_t key_len;
    size_t array_len;
    const char *key;
    void *array;

    /* Private internal fields */
    size_t key_start;
    size_t array_start;
} kaitem_t;

typedef struct {
    int mode;
    size_t num_items;
    kaitem_t *items;
    FILE *file;
    const char *filename;
} kastore_t;

int kastore_open(kastore_t *self, const char *filename, const char *mode, int flags);
int kastore_close(kastore_t *self);

int kastore_get(kastore_t *self, const char *key, kaitem_t **descriptor, int flags);
int kastore_put(kastore_t *self, kaitem_t *item, int flags);

#endif
