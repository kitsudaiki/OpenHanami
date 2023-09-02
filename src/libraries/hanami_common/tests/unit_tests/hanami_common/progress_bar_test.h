/**
 *  @file    statemachine_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef PROGRESSBAR_TEST_H
#define PROGRESSBAR_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class ProgressBar_Test
        : public Hanami::CompareTestHelper
{
public:
    ProgressBar_Test();

private:
    void progress_test();
};

}

#endif // PROGRESSBAR_TEST_H
