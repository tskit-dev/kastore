#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "kastore.h"


static int
kastore_write_header(kastore_t *self)
{
    int ret = 0;
    char header[KAS_HEADER_SIZE];
    uint16_t version_major = KAS_FILE_VERSION_MAJOR;
    uint16_t version_minor = KAS_FILE_VERSION_MINOR;
    uint32_t num_items = (uint32_t) self->num_items;

    memset(header, 0, sizeof(header));
    memcpy(header, KAS_MAGIC, 8);
    memcpy(header + 8, &version_major, 2);
    memcpy(header + 10, &version_minor, 2);
    memcpy(header + 12, &num_items, 4);
    /* Rest of header is reserved */
    if (fwrite(header, KAS_HEADER_SIZE, 1, self->file) != 1) {
        ret = KAS_ERR_IO;
        goto out;
    }

out:
    return ret;
}

static int
kastore_read_header(kastore_t *self)
{
    int ret = 0;
    char header[KAS_HEADER_SIZE];
    size_t count;
    uint16_t version_major, version_minor;
    uint32_t num_items;

    count = fread(header, KAS_HEADER_SIZE, 1, self->file);
    if (count == 0) {
        if (feof(self->file)) {
            ret = KAS_ERR_BAD_FILE_FORMAT;
        } else {
            ret = KAS_ERR_IO;
        }
        goto out;
    }
    if (strncmp(header, KAS_MAGIC, 8) != 0) {
        ret = KAS_ERR_BAD_FILE_FORMAT;
        goto out;
    }
    memcpy(&version_major, header + 8, 2);
    memcpy(&version_minor, header + 10, 2);
    memcpy(&num_items, header + 12, 4);
    self->file_version[0] = (int) version_major;
    self->file_version[1] = (int) version_minor;
    if (self->file_version[0] < KAS_FILE_VERSION_MAJOR) {
        ret = KAS_ERR_VERSION_TOO_OLD;
        goto out;
    } else if (self->file_version[0] > KAS_FILE_VERSION_MAJOR) {
        ret = KAS_ERR_VERSION_TOO_NEW;
        goto out;
    }
    self->num_items = num_items;
out:
    return ret;
}



static int
kastore_write_file(kastore_t *self)
{
    int ret = 0;

    printf("write file\n");
    kastore_print_state(self, stdout);

    ret = kastore_write_header(self);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

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
    if (self->mode == KAS_READ) {
        ret = kastore_read_header(self);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
kastore_close(kastore_t *self)
{
    int ret = 0;
    int err;

    if (self->mode == KAS_WRITE) {
        ret = kastore_write_file(self);
        if (ret != 0) {
            /* Ignore errors on close now */
            fclose(self->file);
            self->file = NULL;
        }
    }
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
kastore_put(kastore_t *self, const char *key, size_t key_len,
       const void *array, size_t array_len, int type, int flags)
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
    new_item->type = type;
    new_item->key = key;
    new_item->key_len = key_len;
    new_item->array = array;
    new_item->array_len = array_len;
out:
    return ret;
}

void
kastore_print_state(kastore_t *self, FILE *out)
{
    kaitem_t *item;
    size_t j;

    fprintf(out, "============================\n");
    fprintf(out, "kastore state\n");
    fprintf(out, "file_version = %d.%d\n", self->file_version[0], self->file_version[1]);
    fprintf(out, "mode = %d\n", self->mode);
    fprintf(out, "num_items = %zu\n", self->num_items);
    fprintf(out, "filename = '%s'\n", self->filename);
    fprintf(out, "file = '%p'\n", (void *) self->file);
    fprintf(out, "============================\n");
    for (j = 0; j < self->num_items; j++) {
        item = self->items + j;
        /* Note this assumes the key is NULL delimited, which is not necessarily true */
        printf("%s: type=%d, key_start=%zu, key_len=%zu, array_start=%zu, array_len=%zu\n",
                item->key, item->type, item->key_start, item->key_len, item->array_start,
                item->array_len);

    }
    fprintf(out, "============================\n");
}
