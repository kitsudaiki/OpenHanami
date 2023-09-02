/**
 *  @file    thread_handler_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef THREADHANDLER_TEST_H
#define THREADHANDLER_TEST_H

#include <unistd.h>

#include <hanami_common/test_helper/compare_test_helper.h>
#include <hanami_common/threading/thread.h>

namespace Kitsunemimi
{

class ThreadHandler_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    ThreadHandler_Test();

private:
    void all_test();
};

//==================================================================================================
// DummyThread for testing
//==================================================================================================
class DummyThread
        : public Kitsunemimi::Thread
{
public:
    DummyThread();

protected:
    void run();
};

}

#endif // THREADHANDLER_TEST_H
