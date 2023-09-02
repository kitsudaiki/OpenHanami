/**
 *  @file    stack_buffer_reserve_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "stack_buffer_reserve_test.h"
#include <hanami_common/buffer/stack_buffer_reserve.h>

namespace Kitsunemimi
{

StackBufferReserve_Test::StackBufferReserve_Test()
    : Kitsunemimi::CompareTestHelper("StackBufferReserve_Test")
{
    addBuffer_test();
    getNumberOfBuffers_test();
    getBuffer_test();
}

/**
 * @brief addStage_test
 */
void
StackBufferReserve_Test::addBuffer_test()
{
    // init
    StackBufferReserve stackBufferReserve;
    DataBuffer* newBuffer = new DataBuffer();

    // run test
    TEST_EQUAL(stackBufferReserve.addBuffer(nullptr), false);
    TEST_EQUAL(stackBufferReserve.addBuffer(newBuffer), true);
}

/**
 * @brief getNumberOfStages_test
 */
void
StackBufferReserve_Test::getNumberOfBuffers_test()
{
    // init
    uint32_t reserveSize = 10;
    StackBufferReserve stackBufferReserve(reserveSize);
    DataBuffer* newBuffer = new DataBuffer();

    // test normal add
    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), 0);
    stackBufferReserve.addBuffer(newBuffer);
    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), 1);

    // test max size
    for(uint32_t i = 0; i < reserveSize+10; i++) {
        stackBufferReserve.addBuffer(new DataBuffer());
    }

    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), reserveSize);
}

/**
 * @brief getStage_test
 */
void
StackBufferReserve_Test::getBuffer_test()
{
    // init
    StackBufferReserve stackBufferReserve;
    DataBuffer* newBuffer = new DataBuffer();
    DataBuffer* returnBuffer = nullptr;
    stackBufferReserve.addBuffer(newBuffer);

    // run test
    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), 1);
    returnBuffer = stackBufferReserve.getBuffer();
    delete returnBuffer;
    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), 0);
    returnBuffer = stackBufferReserve.getBuffer();
    delete returnBuffer;
    TEST_EQUAL(stackBufferReserve.getNumberOfBuffers(), 0);
    returnBuffer = stackBufferReserve.getBuffer();
    delete returnBuffer;
}

}
