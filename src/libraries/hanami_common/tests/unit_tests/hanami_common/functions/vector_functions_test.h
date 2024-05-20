/**
 *  @file    vector_functions_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef VECTOR_functions_TEST_H
#define VECTOR_functions_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class Vectorfunctions_Test : public Hanami::CompareTestHelper
{
   public:
    Vectorfunctions_Test();

   private:
    void removeEmptyStrings_test();
};

}  // namespace Hanami

#endif  // VECTORfunctions_TEST_H
