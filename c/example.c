/* This is a simple program to illustrate the kastore C API. With
 * no arguments it writes a file 'example.kas'. If called with
 * a single command line argument, reads this file in and prints.
 */
#include <err.h>
#include <stdlib.h>
#include <stdio.h>

#include <kastore.h>

static void
handle_kas_error(int line, int retval)
{
    errx(1, "Error at line %d: %s", line, kas_strerror(retval));
}

static void
write_example(const char *path)
{
    int ret;
    kastore_t store;
    const uint32_t a[] = { 1, 2, 3, 4 };
    size_t b_length = 10;
    uint32_t *b = calloc(b_length, sizeof(*b));

    if (b == NULL) {
        err(1, "Out of memory");
    }

    ret = kastore_open(&store, path, "w", 0);
    if (ret != 0) {
        handle_kas_error(__LINE__, ret);
    }
    /* This is the standard 'put' where the library takes a copy of the array.
     * This is the recommended approach unless working with very large arrays. */
    ret = kastore_puts_uint32(&store, "a", a, 4, 0);
    if (ret != 0) {
        handle_kas_error(__LINE__, ret);
    }
    /* This is the own-put variant, where the array is inserted into the store
     * and ownership of the buffer (which must be a pointer returned from
     * malloc/calloc) is passed to the store. The buffer will be freed by the
     * library when 'close' is called. */
    ret = kastore_oputs_uint32(&store, "b", b, b_length, 0);
    if (ret != 0) {
        /* The store only takes ownership of the buffer if oputs succeeds, so
         * we must free b here to avoid leaking memory in error conditions */
        free(b);
        handle_kas_error(__LINE__, ret);
    }

    ret = kastore_close(&store);
    if (ret != 0) {
        handle_kas_error(__LINE__, ret);
    }
}

static void
read_example(const char *path)
{
    int ret;
    kastore_t store;
    uint32_t *array;
    size_t j, k, array_len;
    const char *keys[] = { "a", "b" };

    ret = kastore_open(&store, path, "r", 0);
    if (ret != 0) {
        handle_kas_error(__LINE__, ret);
    }
    for (j = 0; j < sizeof(keys) / sizeof(*keys); j++) {
        ret = kastore_gets_uint32(&store, keys[j], &array, &array_len);
        if (ret != 0) {
            handle_kas_error(__LINE__, ret);
        }
        printf("key: %s = [", keys[j]);
        for (k = 0; k < array_len; k++) {
            printf("%d%s", array[k], k == array_len - 1 ? "]\n" : ", ");
        }
    }
    ret = kastore_close(&store);
    if (ret != 0) {
        handle_kas_error(__LINE__, ret);
    }
}

int
main(int argc, char **argv)
{
    if (argc == 1) {
        write_example("example.kas");
    } else {
        read_example(argv[1]);
    }
    return 0;
}
