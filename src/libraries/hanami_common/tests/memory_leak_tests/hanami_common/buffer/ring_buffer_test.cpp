/**
 *  @file    ring_buffer_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "ring_buffer_test.h"

#include <hanami_common/buffer/ring_buffer.h>

namespace Hanami
{

RingBuffer_Test::RingBuffer_Test()
    : Hanami::MemoryLeakTestHelpter("RingBuffer_Test")
{
    create_delete_test();
}

/**
 * @brief create_delete_test
 */
void
RingBuffer_Test::create_delete_test()
{
    REINIT_TEST();

    RingBuffer*  testBuffer = new RingBuffer();
    delete testBuffer;

    CHECK_MEMORY();
}

}
