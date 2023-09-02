/**
 *  @file    state_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STATE_TEST_H
#define STATE_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class State_Test
        : public Hanami::MemoryLeakTestHelpter
{
public:
    State_Test();

private:
    void create_delete_test();
};

}

#endif // STATE_TEST_H
