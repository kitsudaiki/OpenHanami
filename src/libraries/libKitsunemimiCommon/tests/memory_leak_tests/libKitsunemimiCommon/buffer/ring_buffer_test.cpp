/**
 *  @file    ring_buffer_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "ring_buffer_test.h"

#include <libKitsunemimiCommon/buffer/ring_buffer.h>

namespace Kitsunemimi
{

RingBuffer_Test::RingBuffer_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("RingBuffer_Test")
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

} // namespace Kitsunemimi
