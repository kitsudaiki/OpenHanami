/**
 *  @file    logger_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef LOGGER_TEST_H
#define LOGGER_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class Logger_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    Logger_Test();

private:
    void logger_test();

    void deleteFile(const std::string filePath);
};

} // namespace Kitsunemimi

#endif // LOGGER_TEST_H
