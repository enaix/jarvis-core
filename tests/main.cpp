#include "graph.h"
#include "graph_test.h"

#include <iostream>

int main()
{
    test_graph_init();
    test_graph_access();
    std::cout << "===========" << std::endl << "TESTS PASSED" << std::endl;
    return 0;
}
