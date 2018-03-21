#include "kastore.h"

int
kastore_load(kastore_t *self, const char *filename, int flags)
{
    int ret = 0;

    ret = KAS_ERR_IO;
    return ret;
}

int
kastore_dump(kastore_t *self, const char *filename, int flags)
{
    return 0;
}
