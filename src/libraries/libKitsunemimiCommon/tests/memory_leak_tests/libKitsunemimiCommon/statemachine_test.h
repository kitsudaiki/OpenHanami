/**
 *  @file    statemachine_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STATEMACHINE_TEST_H
#define STATEMACHINE_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{

class Statemachine_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    Statemachine_Test();

private:
    void create_delete_test();
};

} // namespace Kitsunemimi

#endif // STATEMACHINE_TEST_H
