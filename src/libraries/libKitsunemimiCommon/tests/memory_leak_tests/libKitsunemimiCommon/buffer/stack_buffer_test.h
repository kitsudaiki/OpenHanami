/**
 *  @file    stack_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STACK_BUFFER_TEST_H
#define STACK_BUFFER_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{

class StackBuffer_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    StackBuffer_Test();

private:
    void create_delete_test();
};

}

#endif // STACK_BUFFER_TEST_H
