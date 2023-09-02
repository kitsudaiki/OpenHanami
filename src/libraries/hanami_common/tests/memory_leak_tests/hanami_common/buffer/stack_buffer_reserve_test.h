/**
 *  @file    stack_buffer_reserve_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STACK_BUFFER_RESERVE_TEST_H
#define STACK_BUFFER_RESERVE_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class StackBufferReserve_Test
        : public Hanami::MemoryLeakTestHelpter
{
public:
    StackBufferReserve_Test();

private:
    void create_delete_test();
};

}

#endif // STACK_BUFFER_RESERVE_TEST_H
