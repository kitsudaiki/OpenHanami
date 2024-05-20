/**
 *  @file    vector_functions_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "vector_functions_test.h"

#include <hanami_common/functions/vector_functions.h>

namespace Hanami
{

Vectorfunctions_Test::Vectorfunctions_Test() : Hanami::CompareTestHelper("Vectorfunctions_Test")
{
    removeEmptyStrings_test();
}

/**
 * removeEmptyStrings_test
 */
void
Vectorfunctions_Test::removeEmptyStrings_test()
{
    // init
    std::vector<std::string> testVector{"x", "", "y", "z", ""};

    // run task
    removeEmptyStrings(testVector);

    // check result
    TEST_EQUAL(testVector.size(), 3);
    TEST_EQUAL(testVector[0], "x");
    TEST_EQUAL(testVector[1], "y");
    TEST_EQUAL(testVector[2], "z");
}

}  // namespace Hanami
