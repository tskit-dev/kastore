/* Very simple test to check that kastore works with C++. */

#include <iostream>
#include <assert.h>
#include <sstream>

#include <kastore.h>

using namespace std;

void
test_strerror()
{
    std::cout << "test_strerror" << endl;
    std::ostringstream o;
    o << kas_strerror(KAS_ERR_NO_MEMORY);
    assert(std::string("Out of memory").compare(o.str()) == 0);
}

void
test_open_error()
{
    std::cout << "test_open_error" << endl;
    kastore_t store;
    int ret = kastore_open(&store, "no such file", "r", 0);
    assert(ret == KAS_ERR_IO);

    kastore_close(&store);
}

int
main()
{
    test_open_error();
    test_strerror();
    return 0;
}
