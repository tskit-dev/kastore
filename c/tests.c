#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "kastore.h"

#include <CUnit/Basic.h>

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

    /* Read a non existant file */
    ret = kastore_open(&store, "", "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, KAS_ERR_IO);
    CU_ASSERT_EQUAL_FATAL(errno, ENOENT);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* TODO add more */
}

static void
test_simple_round_trip(void)
{
    int ret;
    kastore_t store;
    const uint32_t array[] = {1, 2, 3, 4};
    const uint32_t *a;

    ret = kastore_open(&store, "tmp.kas", "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_put(&store, "c", 1, array, 4, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "b", 1, array, 2, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_put(&store, "a", 1, array, 1, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&store, "tmp.kas", "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL(store.num_items, 3);
    CU_ASSERT_EQUAL(store.items[0].type, KAS_UINT32);
    CU_ASSERT_EQUAL(store.items[0].key_len, 1);
    CU_ASSERT_NSTRING_EQUAL(store.items[0].key, "a", 1);
    CU_ASSERT_EQUAL(store.items[0].array_len, 1);
    a = (const uint32_t *) store.items[0].array;
    CU_ASSERT_EQUAL(a[0], 1);

    CU_ASSERT_EQUAL(store.items[1].type, KAS_UINT32);
    CU_ASSERT_EQUAL(store.items[1].key_len, 1);
    CU_ASSERT_NSTRING_EQUAL(store.items[1].key, "b", 1);
    CU_ASSERT_EQUAL(store.items[1].array_len, 2);
    a = (const uint32_t *) store.items[1].array;
    CU_ASSERT_EQUAL(a[0], 1);
    CU_ASSERT_EQUAL(a[1], 2);

    CU_ASSERT_EQUAL(store.items[2].type, KAS_UINT32);
    CU_ASSERT_EQUAL(store.items[2].key_len, 1);
    CU_ASSERT_NSTRING_EQUAL(store.items[2].key, "c", 1);
    CU_ASSERT_EQUAL(store.items[2].array_len, 4);
    a = (const uint32_t *) store.items[2].array;
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
    return CUE_SUCCESS;
}

static int
kastore_suite_cleanup(void)
{
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
