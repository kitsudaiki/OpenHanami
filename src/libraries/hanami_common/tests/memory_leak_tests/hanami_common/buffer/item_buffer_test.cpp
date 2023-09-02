/**
 *  @file    item_buffer_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "item_buffer_test.h"

#include <hanami_common/buffer/item_buffer.h>

namespace Kitsunemimi
{

ItemBuffer_Test::ItemBuffer_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("ItemBuffer_Test")
{
    create_delete_test();
}

/**
 * @brief create_delete_test
 */
void
ItemBuffer_Test::create_delete_test()
{
    REINIT_TEST();

    ItemBuffer* testBuffer = new ItemBuffer();
    testBuffer->initBuffer<float>(42);
    delete testBuffer;

    CHECK_MEMORY();
}

}
