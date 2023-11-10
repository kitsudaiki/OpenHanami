/**
 *  @file    stack_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STACK_BUFFER_TEST_H
#define STACK_BUFFER_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class StackBuffer_Test : public Hanami::MemoryLeakTestHelpter
{
   public:
    StackBuffer_Test();

   private:
    void create_delete_test();
};

}  // namespace Hanami

#endif  // STACK_BUFFER_TEST_H
