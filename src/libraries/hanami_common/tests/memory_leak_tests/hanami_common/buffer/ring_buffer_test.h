/**
 *  @file    ring_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef RING_BUFFER_TEST_H
#define RING_BUFFER_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class RingBuffer_Test : public Hanami::MemoryLeakTestHelpter
{
   public:
    RingBuffer_Test();

   private:
    void create_delete_test();
};

}  // namespace Hanami

#endif  // RING_BUFFER_TEST_H
