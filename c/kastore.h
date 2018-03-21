#ifndef KASTORE_H
#define KASTORE_H

#define KAS_ERR_GENERIC                               -1
#define KAS_ERR_IO                                    -2


typedef struct {
    int mode;

} kastore_t;

int kastore_load(kastore_t *self, const char *filename, int flags);
int kastore_dump(kastore_t *self, const char *filename, int flags);

#endif
