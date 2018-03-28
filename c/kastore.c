#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "kastore.h"


/* static int */
/* kastore_write_header(kastore_t *self) */
/* { */
/*     int ret = 0; */

/*     return ret; */
/* } */

int
kastore_open(kastore_t *self, const char *filename, const char *mode, int flags)
{
    int ret = 0;

    memset(self, 0, sizeof(*self));
    if (strlen(mode) != 1) {
        ret = KAS_ERR_BAD_MODE;
        goto out;
    }
    if (strncmp(mode, "r", 1) == 0) {
        self->mode = KAS_READ;
    } else if (strncmp(mode, "w", 1) == 0) {
        self->mode = KAS_WRITE;
    } else {
        ret = KAS_ERR_BAD_MODE;
        goto out;
    }
    self->filename = filename;
    self->file = fopen(filename, mode);
    if (self->file == NULL) {
        ret = KAS_ERR_IO;
        goto out;
    }
out:
    return ret;
}

int
kastore_close(kastore_t *self)
{
    int ret = 0;
    int err;

    if (self->file != NULL) {
        err = fclose(self->file);
        if (err != 0) {
            ret = KAS_ERR_IO;
            goto out;
        }
    }
    if (self->items != NULL) {
        free(self->items);
    }
out:
    return ret;
}

int
kastore_get(kastore_t *self, const char *key, kaitem_t **item, int flags)
{
    return 0;
}

int
kastore_put(kastore_t *self, kaitem_t *item, int flags)
{
    int ret = 0;
    kaitem_t *new_item;

    if (self->num_items == 0) {
        self->items = malloc(sizeof(*self->items));
        if (self->items == NULL) {
            ret = KAS_ERR_NO_MEMORY;
            goto out;
        }
    } else {
        assert(1);
    }
    new_item = self->items + self->num_items;
    self->num_items++;
    memset(new_item, 0, sizeof(*new_item));
    new_item->type = item->type;
    new_item->key = item->key;
    new_item->key_len = item->key_len;
    new_item->array = item->array;
    new_item->array_len = item->array_len;
out:
    return ret;
}


