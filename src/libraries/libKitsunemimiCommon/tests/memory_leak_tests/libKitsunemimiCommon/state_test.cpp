/**
 *  @file    state_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "state_test.h"

#include <state.h>

namespace Kitsunemimi
{

State_Test::State_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("State_Test")
{
    create_delete_test();
}

/**
 * @brief create_delete_test
 */
void
State_Test::create_delete_test()
{
    REINIT_TEST();

    State* testState = new State(42, "test-state");
    delete testState;

    CHECK_MEMORY();
}

} // namespace Kitsunemimi
