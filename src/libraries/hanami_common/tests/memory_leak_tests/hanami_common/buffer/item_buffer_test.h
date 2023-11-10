/**
 *  @file    item_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef ITEM_BUFFER_TEST_H
#define ITEM_BUFFER_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class ItemBuffer_Test : public Hanami::MemoryLeakTestHelpter
{
   public:
    ItemBuffer_Test();

   private:
    void create_delete_test();
};

}  // namespace Hanami

#endif  // ITEM_BUFFER_TEST_H
