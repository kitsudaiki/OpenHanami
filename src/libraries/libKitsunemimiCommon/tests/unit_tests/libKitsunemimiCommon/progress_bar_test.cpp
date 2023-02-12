/**
 *  @file    statemachine_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "progress_bar_test.h"

#include <libKitsunemimiCommon/progress_bar.h>

namespace Kitsunemimi
{

ProgressBar_Test::ProgressBar_Test()
    : Kitsunemimi::CompareTestHelper("ProgressBar_Test")
{
    progress_test();
}

/**
 * progress_test
 */
void
ProgressBar_Test::progress_test()
{
    ProgressBar* progressBar = new ProgressBar(100);

    TEST_EQUAL(progressBar->m_maxBarWidth, 100);
    TEST_EQUAL(progressBar->m_progress, 0.0f);

    TEST_EQUAL(progressBar->updateProgress(0.5f), false);
    TEST_EQUAL(progressBar->m_progress, 0.5f);

    TEST_EQUAL(progressBar->updateProgress(1.0f), true);
    TEST_EQUAL(progressBar->m_progress, 1.0f);

    TEST_EQUAL(progressBar->updateProgress(1.5f), true);
    TEST_EQUAL(progressBar->m_progress, 1.0f);

    delete progressBar;
}

}
