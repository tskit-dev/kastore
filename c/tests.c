#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "kastore.h"

#include <CUnit/Basic.h>

char * _tmp_file_name;
FILE * _devnull;

static void
test_bad_open_mode(void)
{
    int ret;
    kastore_t store;
    const char *bad_modes[] = {"", "R", "W", "read", "rw", "write"};
    size_t j;

    for (j = 0; j < sizeof(bad_modes) / sizeof(*bad_modes); j++) {
        ret = kastore_open(&store, "", bad_modes[j], 0);
        CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_MODE);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
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

    /* Read a directory */
    ret = kastore_open(&store, "/", "r", 0);
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
}

static void
test_strerror(void)
{
    printf("FINISH strerror and WRITE TESTS\n");
}

static void
test_bad_types(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = {1};

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_put(&store, "a", 0, array, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_put(&store, "a", 0, array, 1, -2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_put(&store, "a", 0, array, 1, KAS_NUM_TYPES, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);
    ret = kastore_put(&store, "a", 0, array, 1, KAS_NUM_TYPES + 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_BAD_TYPE);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
verify_key_round_trip(const char **keys, size_t num_keys)
{
    int ret;
    kastore_t store;
    size_t j;
    uint32_t array[] = {1};
    uint32_t *a;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < num_keys; j++) {
        ret = kastore_put(&store, keys[j], strlen(keys[j]), array, 1, KAS_UINT32, 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL(store.num_items, num_keys);
    for (j = 0; j < num_keys; j++) {
        ret = kastore_get(&store, keys[j], strlen(keys[j]),
                (const void **) &a, &array_len, &type);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(type, KAS_UINT32);
        CU_ASSERT_EQUAL(array_len, 1);
        CU_ASSERT_EQUAL(a[0], 1);
    }
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_different_key_length(void)
{
    const char *keys[] = {"a", "aa", "aaa", "aaaa", "aaaaa"};
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_different_key_length_reverse(void)
{
    const char *keys[] = {"aaaaaa", "aaaa", "aaa", "aa", "a"};
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_mixed_keys(void)
{
    const char *keys[] = {"x", "aabs", "pqrastuvw", "st", "12345", "67^%"};
    verify_key_round_trip(keys, sizeof(keys) / sizeof(*keys));
}

static void
test_duplicate_key(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = {1};

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
test_empty_key(void)
{
    int ret;
    kastore_t store;
    uint32_t array[] = {1};

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
test_missing_key(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = {1, 2, 3, 4};
    uint32_t *a;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "abc", 3, array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "defg", 4, array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "hijkl", 5, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_get(&store, "xyz", 3, (const void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);
    ret = kastore_get(&store, "a", 1, (const void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);
    ret = kastore_get(&store, "defgh", 5, (const void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_KEY_NOT_FOUND);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}


static void
test_simple_round_trip(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = {1, 2, 3, 4};
    uint32_t *a;
    size_t array_len;
    int type;

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_put(&store, "c", 1, array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "b", 1, array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "a", 1, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL(store.num_items, 3);
    ret = kastore_get(&store, "a", 1, (const void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 1);
    CU_ASSERT_EQUAL(a[0], 1);

    ret = kastore_get(&store, "b", 1, (const void **) &a, &array_len, &type);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(type, KAS_UINT32);
    CU_ASSERT_EQUAL(array_len, 2);
    CU_ASSERT_EQUAL(a[0], 1);
    CU_ASSERT_EQUAL(a[1], 2);

    ret = kastore_get(&store, "c", 1, (const void **) &a, &array_len, &type);
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
    fprintf(stderr, "CUnit error occured: %d: %s\n",
            CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        {"test_bad_open_mode", test_bad_open_mode},
        {"test_open_io_errors", test_open_io_errors},
        {"test_strerror", test_strerror},
        {"test_empty_key", test_empty_key},
        {"test_different_key_length", test_different_key_length},
        {"test_different_key_length_reverse", test_different_key_length_reverse},
        {"test_mixed_keys", test_mixed_keys},
        {"test_duplicate_key", test_duplicate_key},
        {"test_missing_key", test_missing_key},
        {"test_bad_types", test_bad_types},
        {"test_simple_round_trip", test_simple_round_trip},
        CU_TEST_INFO_NULL,
    };

    /* We use initialisers here as the struct definitions change between
     * versions of CUnit */
    CU_SuiteInfo suites[] = {
        {
            .pName = "kastore",
            .pInitFunc = kastore_suite_init,
            .pCleanupFunc = kastore_suite_cleanup,
            .pTests = tests
        },
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
