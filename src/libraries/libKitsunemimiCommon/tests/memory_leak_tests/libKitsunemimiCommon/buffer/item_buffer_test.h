/**
 *  @file    item_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef ITEM_BUFFER_TEST_H
#define ITEM_BUFFER_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{

class ItemBuffer_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    ItemBuffer_Test();

private:
    void create_delete_test();
};

}

#endif // ITEM_BUFFER_TEST_H
