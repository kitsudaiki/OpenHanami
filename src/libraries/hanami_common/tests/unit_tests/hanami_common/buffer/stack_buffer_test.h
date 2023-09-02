/**
 *  @file    stack_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STACK_BUFFER_TEST_H
#define STACK_BUFFER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class StackBuffer_Test
        : public Hanami::CompareTestHelper
{
public:
    StackBuffer_Test();

private:
    void constructor_test();
    void extendBuffer_StackBuffer_test();
    void addData_StackBuffer_test();
    void addObject_StackBuffer_test();
    void getFirstElement_StackBuffer_test();
    void removeFirst_StackBuffer_test();
    void reset_StackBuffer_test();
};

}

#endif // STACK_BUFFER_TEST_H
