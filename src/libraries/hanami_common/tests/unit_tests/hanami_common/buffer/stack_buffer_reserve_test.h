/**
 *  @file    stack_buffer_reserve_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STACK_BUFFER_RESERVE_TEST_H
#define STACK_BUFFER_RESERVE_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class StackBufferReserve_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    StackBufferReserve_Test();

private:
    void addBuffer_test();
    void getNumberOfBuffers_test();
    void getBuffer_test();
};

}

#endif // STACK_BUFFER_RESERVE_TEST_H
