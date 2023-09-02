/**
 *  @file    statemachine_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STATEMACHINE_TEST_H
#define STATEMACHINE_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class Statemachine_Test
        : public Hanami::CompareTestHelper
{
public:
    Statemachine_Test();

private:

    enum states
    {
        SOURCE_STATE = 1,
        TARGET_STATE = 2,
        CHILD_STATE = 3,
        NEXT_STATE = 4,
        GO = 5,
        GOGO = 6,
        FAIL = 7,
    };

    void createNewState_test();
    void setCurrentState_test();
    void addTransition_test();
    void goToNextState_test();
    void getCurrentStateId_test();
    void getCurrentStateName_test();
    void setInitialChildState_test();
    void addChildState_test();
    void isInState_test();
};

}

#endif // STATEMACHINE_TEST_H
