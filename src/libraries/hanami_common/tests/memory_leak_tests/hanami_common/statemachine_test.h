/**
 *  @file    statemachine_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STATEMACHINE_TEST_H
#define STATEMACHINE_TEST_H

#include <hanami_common/test_helper/memory_leak_test_helper.h>

namespace Hanami
{

class Statemachine_Test
        : public Hanami::MemoryLeakTestHelpter
{
public:
    Statemachine_Test();

private:
    void create_delete_test();
};

}

#endif // STATEMACHINE_TEST_H
