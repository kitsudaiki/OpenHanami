/**
 *  @file    vector_methods_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "vector_methods_test.h"

#include <libKitsunemimiCommon/methods/vector_methods.h>

namespace Kitsunemimi
{

VectorMethods_Test::VectorMethods_Test()
    : Kitsunemimi::CompareTestHelper("VectorMethods_Test")
{
    removeEmptyStrings_test();
}

/**
 * removeEmptyStrings_test
 */
void
VectorMethods_Test::removeEmptyStrings_test()
{
    // init
    std::vector<std::string> testVector{"x","","y","z",""};

    // run task
    removeEmptyStrings(testVector);

    // check result
    TEST_EQUAL(testVector.size(), 3);
    TEST_EQUAL(testVector[0], "x");
    TEST_EQUAL(testVector[1], "y");
    TEST_EQUAL(testVector[2], "z");
}

}
