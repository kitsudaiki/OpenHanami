/**
 *  @file      thread_test.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef THREAD_TEST_H
#define THREAD_TEST_H

#include <unistd.h>
#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{

class Thread_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    Thread_Test();

private:
    void create_delete_test();
    void create_delete_with_events_test();
};

}

#endif // THREAD_TEST_H
