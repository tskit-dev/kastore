#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>

#include "kastore.h"

#include <CUnit/Basic.h>

char *_tmp_file_name;
FILE *_devnull;

static void
write_example_file(char *filename)
{
    int ret;
    kastore_t store;
    size_t num_elements = 10;
    uint32_t *array = calloc(num_elements, sizeof(*array));

    ret = kastore_open(&store, filename, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint32(&store, "array", array, num_elements, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_oputs_example(void)
{
    int ret;
    kastore_t store;
    size_t num_elements = 100;
    uint32_t *array = calloc(num_elements, sizeof(*array));

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_oputs_uint32(&store, "big_array", array, num_elements, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_bad_open_flags(void)
{
    int ret;
    kastore_t store;
    const int bad_flags[] = { 4, 5, 1 << 31, -1 };
    size_t j;

    for (j = 0; j < sizeof(bad_flags) / sizeof(*bad_flags); j++) {
        ret = kastore_open(&store, _tmp_file_name, "r", bad_flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_FLAGS);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        ret = kastore_openf(&store, NULL, "r", bad_flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_FLAGS);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_bad_open_mode(void)
{
    int ret;
    kastore_t store;
    const char *bad_modes[] = { "", "R", "W", "read", "rw", "write" };
    size_t j;

    for (j = 0; j < sizeof(bad_modes) / sizeof(*bad_modes); j++) {
        ret = kastore_open(&store, "", bad_modes[j], 0);
        CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_MODE);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        ret = kastore_openf(&store, NULL, bad_modes[j], 0);
        CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_MODE);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }

    /* Additionally, openf doesn't support append mode */
    ret = kastore_openf(&store, NULL, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_MODE);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_openf_bad_file_read_modes(void)
{
    int ret;
    kastore_t store;
    FILE *f;

    write_example_file(_tmp_file_name);
    f = fopen(_tmp_file_name, "w");
    CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
    ret = kastore_openf(&store, f, "r", 0);
    /* Bad file descriptor */
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    fclose(f);

    write_example_file(_tmp_file_name);
    f = fopen(_tmp_file_name, "a");
    CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
    ret = kastore_openf(&store, f, "r", 0);
    /* Bad file descriptor */
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    fclose(f);
}

static void
test_openf_bad_file_write_modes(void)
{
    int ret;
    kastore_t store;
    FILE *f;

    write_example_file(_tmp_file_name);
    f = fopen(_tmp_file_name, "r");
    CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
    ret = kastore_openf(&store, f, "w", 0);
    /* Open succeeds because we don't try to write anything until close */
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    /* Bad file descriptor */
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    fclose(f);
}

static void
test_open_io_errors(void)
{
    int ret;
    kastore_t store;
    const char *msg1, *msg2;

    /* Read a non existent file */
    ret = kastore_open(&store, "", "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, ENOENT);
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    CU_ASSERT_STRING_EQUAL(msg1, msg2);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Append a non existent file */
    ret = kastore_open(&store, "", "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, ENOENT);
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    CU_ASSERT_STRING_EQUAL(msg1, msg2);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Read a directory */
    ret = kastore_open(&store, "/", "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, EISDIR)
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Append a directory */
    ret = kastore_open(&store, "/", "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, EISDIR)
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Write a directory */
    ret = kastore_open(&store, "./", "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, EISDIR)
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Write without permissions */
    ret = kastore_open(&store, "/noway.kas", "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    msg1 = kas_strerror(KAS_ERR_IO);
    msg2 = strerror(errno);
    CU_ASSERT_EQUAL_FATAL(errno, EACCES);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Reading /dev/null returns 0 bytes */
    ret = kastore_open(&store, "/dev/null", "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_EOF);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Appending /dev/null gives an error. */
    ret = kastore_open(&store, "/dev/null", "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_EOF);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_append_empty_file(void)
{
    int ret;
    kastore_t store;
    FILE *f;

    f = fopen(_tmp_file_name, "w");
    CU_ASSERT_FATAL(f != NULL);
    fclose(f);

    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_EOF);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_write_errors(void)
{
    int ret;
    kastore_t store;
    int64_t a[4] = { 1, 2, 3, 4 };

    /* Write /dev/null should be fine */
    ret = kastore_open(&store, "/dev/random", "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", a, 4, KAS_INT64, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", a, 4, KAS_INT64, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* TODO find some way to make it so we get IO errors when we flush */
}

static void
test_strerror(void)
{
    const int max_err = 100; /* arbitrary */
    int err;
    const char *str;

    /* Make sure the errno=0 codepath for IO errors is exercised */
    errno = 0;
    for (err = 1; err < max_err; err++) {
        str = kas_strerror(-err);
        CU_ASSERT_NOT_EQUAL_FATAL(str, NULL);
        CU_ASSERT(strlen(str) > 0);
    }
}

static void
test_bad_types(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = { 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "a", array, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_puts(&store, "a", array, 1, -2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_puts(&store, "a", array, 1, KAS_NUM_TYPES, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_puts(&store, "a", array, 1, KAS_NUM_TYPES + 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);

    ret = kastore_oputs(&store, "a", NULL, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_oputs(&store, "a", NULL, 1, -2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_oputs(&store, "a", NULL, 1, KAS_NUM_TYPES, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_oputs(&store, "a", NULL, 1, KAS_NUM_TYPES + 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
verify_key_round_trip(const char **keys, size_t num_keys)
{
    int ret;
    kastore_t store;
    size_t j, m;
    uint32_t array[] = { 1 };
    uint32_t *a;
    size_t array_len;
    int type;
    const char *modes[] = { "a", "w" };

    for (m = 0; m < 2; m++) {
        if (m == 0) {
            /* Write an empty kastore first so we can append to it */
            ret = kastore_open(&store, _tmp_file_name, "w", 0);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            ret = kastore_close(&store);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
        }
        ret = kastore_open(&store, _tmp_file_name, modes[m], 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        for (j = 0; j < num_keys; j++) {
            ret = kastore_put(&store, keys[j], strlen(keys[j]), array, 1, KAS_UINT32, 0);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
        }
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        ret = kastore_open(&store, _tmp_file_name, "r", 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        kastore_print_state(&store, _devnull);

        CU_ASSERT_EQUAL(store.num_items, num_keys);
        for (j = 0; j < num_keys; j++) {
            ret = kastore_get(
                &store, keys[j], strlen(keys[j]), (void **) &a, &array_len, &type);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL(type, KAS_UINT32);
            CU_ASSERT_EQUAL(array_len, 1);
            CU_ASSERT_EQUAL(a[0], 1);
        }
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_different_key_length(void)
{
    const char *keys[] = { "a", "aa", "aaa", "aaaa", "aaaaa" };
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_different_key_length_reverse(void)
{
    const char *keys[] = { "aaaaaa", "aaaa", "aaa", "aa", "a" };
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_mixed_keys(void)
{
    const char *keys[] = { "x", "aabs", "pqrastuvw", "st", "12345", "67^%" };
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_put_copy_array(void)
{
    kastore_t store;
    size_t array_len = 10;
    size_t read_array_len;
    uint32_t array[array_len], *read_array;
    int ret, type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(array, 0, sizeof(array));
    ret = kastore_puts_uint32(&store, "a", array, array_len, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(array, 0xff, sizeof(array));
    ret = kastore_puts_uint32(&store, "b", array, array_len, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(array, 0, sizeof(array));
    ret = kastore_gets(&store, "a", (void **) &read_array, &read_array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL_FATAL(read_array_len, array_len);
    CU_ASSERT_TRUE(memcmp(array, read_array, sizeof(array)) == 0);

    memset(array, 0xff, sizeof(array));
    ret = kastore_gets(&store, "b", (void **) &read_array, &read_array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL_FATAL(read_array_len, array_len);
    CU_ASSERT_TRUE(memcmp(array, read_array, sizeof(array)) == 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_duplicate_key(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = { 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_put(&store, "a", 1, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "b", 1, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "a", 1, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_DUPLICATE_KEY);
    CU_ASSERT_EQUAL_FATAL(store.num_items, 2);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(store.num_items, 2);
    ret = kastore_close(&store);
}

static void
test_duplicate_key_oput(void)
{
    int ret;
    kastore_t store;
    uint32_t *a = calloc(1, sizeof(uint32_t));
    uint32_t *b = calloc(1, sizeof(uint32_t));
    uint32_t *c = calloc(1, sizeof(uint32_t));

    CU_ASSERT_FATAL(a != NULL && b != NULL && c != NULL);

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_oput(&store, "a", 1, a, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oput(&store, "b", 1, b, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oput(&store, "a", 1, c, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_DUPLICATE_KEY);
    CU_ASSERT_EQUAL_FATAL(store.num_items, 2);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* a and b were inserted successfully so we don't need to free then. c
     * was not, so it won't be freed */
    free(c);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(store.num_items, 2);
    ret = kastore_close(&store);
}

static void
test_bad_flag_put(void)
{
    int ret;
    kastore_t store;
    uint32_t *a = calloc(1, sizeof(uint32_t));

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_put(&store, "a", 1, a, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "b", 1, a, 1, KAS_UINT32, 5);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_FLAGS);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    kas_safe_free(a);
}

static void
test_bad_flag_oput(void)
{
    int ret;
    kastore_t store;
    uint32_t *a = calloc(1, sizeof(uint32_t));

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_oput(&store, "a", 1, a, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oput(&store, "b", 1, a, 1, KAS_UINT32, 1);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_FLAGS);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_empty_key(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = { 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "", 0, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_EMPTY_KEY);
    ret = kastore_put(&store, "b", 0, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_EMPTY_KEY);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_put_read_mode(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = { 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_ILLEGAL_OPERATION);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_get_write_mode(void)
{
    int ret;
    kastore_t store;
    uint32_t *a;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets(&store, "xyz", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_ILLEGAL_OPERATION);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets(&store, "xyz", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_ILLEGAL_OPERATION);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_contains(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "abc", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* It's an error to query a store in write mode */
    CU_ASSERT_EQUAL(kastore_containss(&store, "abc"), KAS_ERR_ILLEGAL_OPERATION);
    CU_ASSERT_EQUAL(kastore_containss(&store, "xyz"), KAS_ERR_ILLEGAL_OPERATION);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "abc"), 1);
    CU_ASSERT_EQUAL_FATAL(kastore_contains(&store, "abc", 3), 1);
    CU_ASSERT_EQUAL_FATAL(kastore_contains(&store, "abc", 2), 0);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "ab"), 0);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "xyz"), 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_missing_key(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "abc", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "defg", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "hijkl", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets(&store, "xyz", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);
    ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);
    ret = kastore_gets(&store, "defgh", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "xyz"), 0);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "a"), 0);
    CU_ASSERT_EQUAL_FATAL(kastore_containss(&store, "defgh"), 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_simple_round_trip(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a;
    size_t j, array_len;
    int type;
    int flags[] = { 0, 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "c", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        ret = kastore_open(&store, _tmp_file_name, "r", flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(store.num_items, 3);
        ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);
        CU_ASSERT_EQUAL(a[0], 1);

        ret = kastore_gets(&store, "b", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);

        ret = kastore_gets(&store, "c", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 4);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);
        CU_ASSERT_EQUAL(a[2], 3);
        CU_ASSERT_EQUAL(a[3], 4);

        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_take_ownership(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a, *b, *b2, *c;
    size_t j, array_len;
    int type;
    int flags[] = { 0, 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "c", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "not_got", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        ret = kastore_open(
            &store, _tmp_file_name, "r", flags[j] | KAS_GET_TAKES_OWNERSHIP);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(store.num_items, 4);
        ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);

        ret = kastore_gets(&store, "b", (void **) &b, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);
        CU_ASSERT_EQUAL(b[0], 1);
        CU_ASSERT_EQUAL(b[1], 2);
        kas_safe_free(b);
        /* Read twice to check that kastore gives a new buffer */
        ret = kastore_gets(&store, "b", (void **) &b2, &array_len, &type);
        CU_ASSERT_NOT_EQUAL(b, b2);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);

        ret = kastore_gets(&store, "c", (void **) &c, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 4);

        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(b2[0], 1);
        CU_ASSERT_EQUAL(b2[1], 2);
        CU_ASSERT_EQUAL(c[0], 1);
        CU_ASSERT_EQUAL(c[1], 2);
        CU_ASSERT_EQUAL(c[2], 3);
        CU_ASSERT_EQUAL(c[3], 4);

        kas_safe_free(a);
        kas_safe_free(b2);
        kas_safe_free(c);
    }
}

static void
test_borrow_array(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a, *b, *c, *d;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "c", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", array, 2, KAS_UINT32, KAS_BORROWS_ARRAY);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint32(&store, "a", array, 1, KAS_BORROWS_ARRAY);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "d", NULL, 0, KAS_UINT32, KAS_BORROWS_ARRAY);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL(store.num_items, 4);
    ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 1);

    ret = kastore_gets(&store, "b", (void **) &b, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 2);
    CU_ASSERT_EQUAL(b[0], 1);
    CU_ASSERT_EQUAL(b[1], 2);

    ret = kastore_gets(&store, "c", (void **) &c, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 4);

    ret = kastore_gets(&store, "d", (void **) &d, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
verify_simple_file_round_trip(const char *write_mode, const char *read_mode)
{
    int ret;
    FILE *f;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a;
    size_t j, array_len;
    int type;
    int flags[] = { 0, 1 };

    f = fopen(_tmp_file_name, write_mode);
    CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
    ret = kastore_openf(&store, f, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_puts(&store, "c", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    fclose(f);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        f = fopen(_tmp_file_name, read_mode);
        CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
        ret = kastore_openf(&store, f, "r", flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(store.num_items, 3);
        ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);
        CU_ASSERT_EQUAL(a[0], 1);

        ret = kastore_gets(&store, "b", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);

        ret = kastore_gets(&store, "c", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 4);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);
        CU_ASSERT_EQUAL(a[2], 3);
        CU_ASSERT_EQUAL(a[3], 4);

        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        fclose(f);
    }

    /* Truncate the file to make sure there's no side effects. */
    f = fopen(_tmp_file_name, "w");
    CU_ASSERT_NOT_EQUAL_FATAL(f, NULL);
    fclose(f);
}

static void
test_simple_round_trip_file_modes(void)
{
    /* Various open modes will work fine. Simpler to test they work than
     * to rule then out. */
    verify_simple_file_round_trip("w", "r");
    verify_simple_file_round_trip("w", "rw");
    verify_simple_file_round_trip("w+", "r");
    verify_simple_file_round_trip("a", "rw");
    verify_simple_file_round_trip("a", "r+");
    verify_simple_file_round_trip("a+", "rw");
}

static void
test_simple_round_trip_zero_keys(void)
{
    int ret;
    kastore_t store;
    size_t j;
    int flags[] = { 0, 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        ret = kastore_open(&store, _tmp_file_name, "r", flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(store.num_items, 0);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_simple_round_trip_oput_buffers(void)
{
    int ret;
    kastore_t store;
    uint32_t *array_a = malloc(1 * sizeof(uint32_t));
    uint32_t *array_b = malloc(2 * sizeof(uint32_t));
    uint32_t *array_c = malloc(4 * sizeof(uint32_t));
    uint32_t *a;
    size_t j, array_len;
    int type;
    int flags[] = { 0, 1 };

    CU_ASSERT_FATAL(array_a != NULL && array_b != NULL && array_c != NULL);
    array_a[0] = 1;
    array_b[0] = 1;
    array_b[1] = 2;
    array_c[0] = 1;
    array_c[1] = 2;
    array_c[2] = 3;
    array_c[3] = 4;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint32(&store, "c", array_c, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint32(&store, "b", array_b, 2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint32(&store, "a", array_a, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        ret = kastore_open(&store, _tmp_file_name, "r", flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(store.num_items, 3);
        ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);
        CU_ASSERT_EQUAL(a[0], 1);

        ret = kastore_gets(&store, "b", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);

        ret = kastore_gets(&store, "c", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 4);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);
        CU_ASSERT_EQUAL(a[2], 3);
        CU_ASSERT_EQUAL(a[3], 4);

        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_simple_round_trip_append(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = { 1, 2, 3, 4 };
    uint32_t *a;
    size_t j, array_len;
    int type;
    int flags[] = { 0, 1 };

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "c", array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "b", array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "a", array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, "d", NULL, 0, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(flags) / sizeof(*flags); j++) {
        ret = kastore_open(&store, _tmp_file_name, "r", flags[j]);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        CU_ASSERT_EQUAL(store.num_items, 4);
        ret = kastore_gets(&store, "a", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_TRUE(kastore_containss(&store, "a"));

        ret = kastore_gets(&store, "b", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 2);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);
        CU_ASSERT_TRUE(kastore_containss(&store, "b"));

        ret = kastore_gets(&store, "c", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 4);
        CU_ASSERT_EQUAL(a[0], 1);
        CU_ASSERT_EQUAL(a[1], 2);
        CU_ASSERT_EQUAL(a[2], 3);
        CU_ASSERT_EQUAL(a[3], 4);
        CU_ASSERT_TRUE(kastore_containss(&store, "c"));

        ret = kastore_gets(&store, "d", (void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 0);
        CU_ASSERT_TRUE(kastore_containss(&store, "d"));

        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_gets_type_errors(void)
{
    int ret;
    kastore_t store;
    int8_t a = INT8_MAX;
    int8_t *read;
    uint8_t *readu8;
    double *readf64;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int8(&store, "a", &a, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_int8(&store, "abcdefg", &read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);

    ret = kastore_gets_int8(&store, "a", &read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(read[0], a);

    ret = kastore_gets_uint8(&store, "a", &readu8, &len);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_TYPE_MISMATCH);

    ret = kastore_gets_float64(&store, "a", &readf64, &len);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_TYPE_MISMATCH);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_int8(void)
{
    int ret;
    kastore_t store;
    int8_t max = INT8_MAX;
    int8_t min = INT8_MIN;
    int8_t *zero = calloc(1, sizeof(*zero));
    int8_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int8(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int8(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_int8(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_int8(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_int8(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_int8(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_uint8(void)
{
    int ret;
    kastore_t store;
    uint8_t max = UINT8_MAX;
    uint8_t min = 0;
    uint8_t *zero = calloc(1, sizeof(*zero));
    uint8_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint8(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint8(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint8(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_uint8(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_uint8(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_uint8(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_int16(void)
{
    int ret;
    kastore_t store;
    int16_t max = INT16_MAX;
    int16_t min = INT16_MIN;
    int16_t *zero = calloc(1, sizeof(*zero));
    int16_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int16(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int16(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_int16(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_int16(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_int16(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_int16(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_uint16(void)
{
    int ret;
    kastore_t store;
    uint16_t max = UINT16_MAX;
    uint16_t min = 0;
    uint16_t *zero = calloc(1, sizeof(*zero));
    uint16_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint16(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint16(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint16(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_uint16(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_uint16(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_uint16(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_int32(void)
{
    int ret;
    kastore_t store;
    int32_t max = INT32_MAX;
    int32_t min = INT32_MIN;
    int32_t *zero = calloc(1, sizeof(*zero));
    int32_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int32(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int32(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_int32(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_int32(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_int32(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_int32(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_uint32(void)
{
    int ret;
    kastore_t store;
    uint32_t max = UINT32_MAX;
    uint32_t min = 0;
    uint32_t *zero = calloc(1, sizeof(*zero));
    uint32_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint32(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint32(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint32(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_uint32(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_uint32(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_uint32(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_int64(void)
{
    int ret;
    kastore_t store;
    int64_t max = INT64_MAX;
    int64_t min = INT64_MIN;
    int64_t *zero = calloc(1, sizeof(*zero));
    int64_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int64(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_int64(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_int64(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_int64(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_int64(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_int64(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_uint64(void)
{
    int ret;
    kastore_t store;
    uint64_t max = UINT64_MAX;
    uint64_t min = 0;
    uint64_t *zero = calloc(1, sizeof(*zero));
    uint64_t *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint64(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_uint64(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_uint64(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_uint64(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_uint64(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_uint64(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_float32(void)
{
    int ret;
    kastore_t store;
    float max = FLT_MAX;
    float min = FLT_MIN;
    float *zero = calloc(1, sizeof(*zero));
    float *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_float32(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_float32(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_float32(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_float32(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_float32(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_float32(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_round_trip_float64(void)
{
    int ret;
    kastore_t store;
    double max = DBL_MAX;
    double min = DBL_MIN;
    double *zero = calloc(1, sizeof(*zero));
    double *max_read, *min_read, *zero_read;
    size_t len;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_float64(&store, "min", &min, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts_float64(&store, "max", &max, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_oputs_float64(&store, "zero", zero, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_gets_float64(&store, "min", &min_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(min_read[0], min);
    ret = kastore_gets_float64(&store, "max", &max_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(max_read[0], max);
    ret = kastore_gets_float64(&store, "zero", &zero_read, &len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(len, 1);
    CU_ASSERT_EQUAL_FATAL(zero_read[0], 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
verify_bad_file(const char *filename, int err)
{
    int ret;
    kastore_t store;

    ret = kastore_open(&store, filename, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, err);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, filename, "r", KAS_READ_ALL);
    CU_ASSERT_EQUAL_FATAL(ret, err);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_short_header(void)
{
    FILE *f;
    char zeros[KAS_HEADER_SIZE];
    size_t j;

    memset(zeros, 0, sizeof(zeros));

    for (j = 1; j < KAS_HEADER_SIZE; j++) {
        f = fopen(_tmp_file_name, "w");
        CU_ASSERT_EQUAL_FATAL(fwrite(zeros, 1, j, f), j);
        fclose(f);
        verify_bad_file(_tmp_file_name, KAS_ERR_BAD_FILE_FORMAT);
    }
}

static void
test_empty_file(void)
{
    verify_bad_file("test-data/malformed/empty_file.kas", KAS_ERR_EOF);
}

static void
test_read_bad_types(void)
{
    verify_bad_file("test-data/malformed/bad_type_11.kas", KAS_ERR_BAD_TYPE);
    verify_bad_file("test-data/malformed/bad_type_20.kas", KAS_ERR_BAD_TYPE);
}

static void
test_bad_filesizes(void)
{
    verify_bad_file(
        "test-data/malformed/bad_filesize_0_-1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file("test-data/malformed/bad_filesize_0_1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_filesize_0_1024.kas", KAS_ERR_BAD_FILE_FORMAT);

    verify_bad_file(
        "test-data/malformed/bad_filesize_10_-1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_filesize_10_1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_filesize_10_1024.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_bad_magic_number(void)
{
    verify_bad_file("test-data/malformed/bad_magic_number.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_version_0(void)
{
    verify_bad_file("test-data/malformed/version_0.kas", KAS_ERR_VERSION_TOO_OLD);
}

static void
test_version_100(void)
{
    verify_bad_file("test-data/malformed/version_100.kas", KAS_ERR_VERSION_TOO_NEW);
}

static void
test_truncated_file(void)
{
    verify_bad_file("test-data/malformed/truncated_file.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_key_offset_outside_file(void)
{
    verify_bad_file(
        "test-data/malformed/key_offset_outside_file.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_array_offset_outside_file(void)
{
    verify_bad_file(
        "test-data/malformed/array_offset_outside_file.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_key_len_outside_file(void)
{
    verify_bad_file(
        "test-data/malformed/key_len_outside_file.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_array_len_outside_file(void)
{
    verify_bad_file(
        "test-data/malformed/array_len_outside_file.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_bad_key_start(void)
{
    verify_bad_file("test-data/malformed/bad_key_start_-1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file("test-data/malformed/bad_key_start_1.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_bad_array_start(void)
{
    verify_bad_file(
        "test-data/malformed/bad_array_start_-8.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_array_start_-1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_array_start_1.kas", KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file(
        "test-data/malformed/bad_array_start_8.kas", KAS_ERR_BAD_FILE_FORMAT);
}

static void
test_truncated_file_correct_size(void)
{
    verify_bad_file("test-data/malformed/truncated_file_correct_size_100.kas",
        KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file("test-data/malformed/truncated_file_correct_size_128.kas",
        KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file("test-data/malformed/truncated_file_correct_size_129.kas",
        KAS_ERR_BAD_FILE_FORMAT);
    verify_bad_file("test-data/malformed/truncated_file_correct_size_200.kas",
        KAS_ERR_BAD_FILE_FORMAT);
}

static void
verify_all_types_n_elements(size_t n)
{
    int ret;
    kastore_t store;
    const char *filename_pattern = "test-data/v1/all_types_%d_elements.kas";
    const char *keys[] = { "uint8", "int8", "uint16", "int16", "uint32", "int32",
        "uint64", "int64", "float32", "float64" };
    const int types[] = { KAS_UINT8, KAS_INT8, KAS_UINT16, KAS_INT16, KAS_UINT32,
        KAS_INT32, KAS_UINT64, KAS_INT64, KAS_FLOAT32, KAS_FLOAT64 };
    size_t j, k, array_len;
    void *a;
    int type;
    char filename[1024];

    sprintf(filename, filename_pattern, n);

    ret = kastore_open(&store, filename, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < sizeof(keys) / sizeof(*keys); j++) {
        ret = kastore_gets(&store, keys[j], &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(array_len, n);
        CU_ASSERT_FATAL(a != NULL);
        CU_ASSERT_EQUAL_FATAL(type, types[j]);
        for (k = 0; k < array_len; k++) {
            switch (type) {
                case KAS_UINT8:
                    CU_ASSERT_EQUAL_FATAL(((uint8_t *) a)[k], (uint8_t) k);
                    break;
                case KAS_INT8:
                    CU_ASSERT_EQUAL_FATAL(((int8_t *) a)[k], (int8_t) k);
                    break;
                case KAS_UINT16:
                    CU_ASSERT_EQUAL_FATAL(((uint16_t *) a)[k], (uint16_t) k);
                    break;
                case KAS_INT16:
                    CU_ASSERT_EQUAL_FATAL(((int16_t *) a)[k], (int16_t) k);
                    break;
                case KAS_UINT32:
                    CU_ASSERT_EQUAL_FATAL(((uint32_t *) a)[k], (uint32_t) k);
                    break;
                case KAS_INT32:
                    CU_ASSERT_EQUAL_FATAL(((int32_t *) a)[k], (int32_t) k);
                    break;
                case KAS_UINT64:
                    CU_ASSERT_EQUAL_FATAL(((uint64_t *) a)[k], (uint64_t) k);
                    break;
                case KAS_INT64:
                    CU_ASSERT_EQUAL_FATAL(((int64_t *) a)[k], (int64_t) k);
                    break;
                case KAS_FLOAT32:
                    CU_ASSERT_EQUAL_FATAL(((float *) a)[k], (float) k);
                    break;
                case KAS_FLOAT64:
                    CU_ASSERT_EQUAL_FATAL(((double *) a)[k], (double) k);
                    break;
            }
        }
    }
    kastore_close(&store);
}

static void
test_all_types_n_elements(void)
{
    size_t j;

    for (j = 0; j < 10; j++) {
        verify_all_types_n_elements(j);
    }
}

static void
test_library_version(void)
{
    kas_version_t version = kas_version();

    CU_ASSERT_EQUAL_FATAL(version.major, KAS_VERSION_MAJOR);
    CU_ASSERT_EQUAL_FATAL(version.minor, KAS_VERSION_MINOR);
    CU_ASSERT_EQUAL_FATAL(version.patch, KAS_VERSION_PATCH);
}

static void
test_meson_version(void)
{
    char version[100];

    sprintf(
        version, "%d.%d.%d", KAS_VERSION_MAJOR, KAS_VERSION_MINOR, KAS_VERSION_PATCH);
    /* the MESON_VERSION define is passed in by meson when compiling */
    CU_ASSERT_STRING_EQUAL(version, MESON_VERSION);
}

/*=================================================
  Test suite management
  =================================================
*/

static int
kastore_suite_init(void)
{
    int fd;
    static char template[] = "/tmp/kas_c_test_XXXXXX";

    _tmp_file_name = NULL;
    _devnull = NULL;

    _tmp_file_name = malloc(sizeof(template));
    if (_tmp_file_name == NULL) {
        return CUE_NOMEMORY;
    }
    strcpy(_tmp_file_name, template);
    fd = mkstemp(_tmp_file_name);
    if (fd == -1) {
        return CUE_SINIT_FAILED;
    }
    close(fd);
    _devnull = fopen("/dev/null", "w");
    if (_devnull == NULL) {
        return CUE_SINIT_FAILED;
    }
    return CUE_SUCCESS;
}

static int
kastore_suite_cleanup(void)
{
    if (_tmp_file_name != NULL) {
        unlink(_tmp_file_name);
        free(_tmp_file_name);
    }
    if (_devnull != NULL) {
        fclose(_devnull);
    }
    return CUE_SUCCESS;
}

static void
handle_cunit_error(void)
{
    fprintf(stderr, "CUnit error occured: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        { "test_oputs_example", test_oputs_example },
        { "test_bad_open_flags", test_bad_open_flags },
        { "test_bad_open_mode", test_bad_open_mode },
        { "test_openf_bad_file_read_modes", test_openf_bad_file_read_modes },
        { "test_openf_bad_file_write_modes", test_openf_bad_file_write_modes },
        { "test_open_io_errors", test_open_io_errors },
        { "test_append_empty_file", test_append_empty_file },
        { "test_write_errors", test_write_errors },
        { "test_strerror", test_strerror },
        { "test_empty_key", test_empty_key },
        { "test_get_write_mode", test_get_write_mode },
        { "test_put_read_mode", test_put_read_mode },
        { "test_different_key_length", test_different_key_length },
        { "test_different_key_length_reverse", test_different_key_length_reverse },
        { "test_mixed_keys", test_mixed_keys },
        { "test_put_copy_array", test_put_copy_array },
        { "test_duplicate_key", test_duplicate_key },
        { "test_duplicate_key_oput", test_duplicate_key_oput },
        { "test_bad_flag_oput", test_bad_flag_oput },
        { "test_bad_flag_put", test_bad_flag_put },
        { "test_missing_key", test_missing_key },
        { "test_contains", test_contains },
        { "test_bad_types", test_bad_types },
        { "test_simple_round_trip", test_simple_round_trip },
        { "test_simple_round_trip_file_modes", test_simple_round_trip_file_modes },
        { "test_simple_round_trip_zero_keys", test_simple_round_trip_zero_keys },
        { "test_simple_round_trip_oput_buffers", test_simple_round_trip_oput_buffers },
        { "test_simple_round_trip_append", test_simple_round_trip_append },
        { "test_gets_type_errors", test_gets_type_errors },
        { "test_round_trip_int8", test_round_trip_int8 },
        { "test_round_trip_uint8", test_round_trip_uint8 },
        { "test_round_trip_int16", test_round_trip_int16 },
        { "test_round_trip_uint16", test_round_trip_uint16 },
        { "test_round_trip_int32", test_round_trip_int32 },
        { "test_round_trip_uint32", test_round_trip_uint32 },
        { "test_round_trip_int64", test_round_trip_int64 },
        { "test_round_trip_uint64", test_round_trip_uint64 },
        { "test_round_trip_float32", test_round_trip_float32 },
        { "test_round_trip_float64", test_round_trip_float64 },
        { "test_short_header", test_short_header },
        { "test_empty_file", test_empty_file },
        { "test_read_bad_types", test_read_bad_types },
        { "test_bad_filesizes", test_bad_filesizes },
        { "test_bad_magic_number", test_bad_magic_number },
        { "test_version_0", test_version_0 },
        { "test_version_100", test_version_100 },
        { "test_truncated_file", test_truncated_file },
        { "test_key_offset_outside_file", test_key_offset_outside_file },
        { "test_array_offset_outside_file", test_array_offset_outside_file },
        { "test_key_len_outside_file", test_key_len_outside_file },
        { "test_array_len_outside_file", test_array_len_outside_file },
        { "test_bad_key_start", test_bad_key_start },
        { "test_bad_array_start", test_bad_array_start },
        { "test_truncated_file_correct_size", test_truncated_file_correct_size },
        { "test_all_types_n_elements", test_all_types_n_elements },
        { "test_library_version", test_library_version },
        { "test_meson_version", test_meson_version },
        { "test_take_ownership", test_take_ownership },
        { "test_borrow_array", test_borrow_array },
        CU_TEST_INFO_NULL,
    };

    /* We use initialisers here as the struct definitions change between
     * versions of CUnit */
    CU_SuiteInfo suites[] = {
        { .pName = "kastore",
            .pInitFunc = kastore_suite_init,
            .pCleanupFunc = kastore_suite_cleanup,
            .pTests = tests },
        CU_SUITE_INFO_NULL,
    };
    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    if (CUE_SUCCESS != CU_register_suites(suites)) {
        handle_cunit_error();
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);

    if (argc == 1) {
        CU_basic_run_tests();
    } else if (argc == 2) {
        suite = CU_get_suite_by_name("kastore", CU_get_registry());
        if (suite == NULL) {
            printf("Suite not found\n");
            return EXIT_FAILURE;
        }
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
            return EXIT_FAILURE;
        }
        CU_basic_run_test(suite, test);
    } else {
        printf("usage: ./tests <test_name>\n");
        return EXIT_FAILURE;
    }

    ret = EXIT_SUCCESS;
    if (CU_get_number_of_tests_failed() != 0) {
        printf("Test failed!\n");
        ret = EXIT_FAILURE;
    }
    CU_cleanup_registry();
    return ret;
}
