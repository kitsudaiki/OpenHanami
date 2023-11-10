/**
 *  @file      thread_test.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef THREAD_TEST_H
#define THREAD_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>
#include <unistd.h>

namespace Hanami
{

class Thread_Test : public Hanami::MemoryLeakTestHelpter
{
   public:
    Thread_Test();

   private:
    void create_delete_test();
    void create_delete_with_events_test();
};

}  // namespace Hanami

#endif  // THREAD_TEST_H
