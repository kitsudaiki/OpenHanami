/**
 *  @file    state_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STATE_TEST_H
#define STATE_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class State_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    State_Test();

private:
    enum states
    {
        SOURCE_STATE = 1,
        TARGET_STATE = 2,
        CHILD_STATE = 3,
        NEXT_STATE = 4,
        INITIAL_STATE = 5,
        GO = 6,
        GOGO = 7,
        FAIL = 8,
    };

    void addTransition_test();
    void next_test();
    void setInitialChildState_test();
    void addChildState_test();
};

}

#endif // STATE_TEST_H
