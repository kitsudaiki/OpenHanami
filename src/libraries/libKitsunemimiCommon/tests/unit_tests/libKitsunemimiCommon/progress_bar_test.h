/**
 *  @file    statemachine_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef PROGRESSBAR_TEST_H
#define PROGRESSBAR_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class ProgressBar_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    ProgressBar_Test();

private:
    void progress_test();
};

} // namespace Kitsunemimi

#endif // PROGRESSBAR_TEST_H
