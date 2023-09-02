/**
 *  @file    vector_methods_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef VECTOR_METHODS_TEST_H
#define VECTOR_METHODS_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class VectorMethods_Test
        : public Hanami::CompareTestHelper
{
public:
    VectorMethods_Test();

private:
    void removeEmptyStrings_test();
};

}

#endif // VECTORMETHODS_TEST_H
