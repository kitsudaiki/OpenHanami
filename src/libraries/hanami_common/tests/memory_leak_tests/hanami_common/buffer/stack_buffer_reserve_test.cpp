/**
 *  @file    stack_buffer_reserve_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "stack_buffer_reserve_test.h"

#include <hanami_common/buffer/stack_buffer_reserve.h>

namespace Hanami
{

StackBufferReserve_Test::StackBufferReserve_Test()
    : Hanami::MemoryLeakTestHelpter("StackBufferReserve_Test")
{
    create_delete_test();
}

/**
 * @brief create_delete_test
 */
void
StackBufferReserve_Test::create_delete_test()
{
    REINIT_TEST();

    StackBufferReserve* testBuffer = new StackBufferReserve();
    DataBuffer* input = new DataBuffer();
    testBuffer->addBuffer(input);
    delete testBuffer;

    CHECK_MEMORY();
}

}  // namespace Hanami
