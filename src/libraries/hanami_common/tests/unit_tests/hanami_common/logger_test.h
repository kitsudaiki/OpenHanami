/**
 *  @file    logger_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef LOGGER_TEST_H
#define LOGGER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class Logger_Test
        : public Hanami::CompareTestHelper
{
public:
    Logger_Test();

private:
    void logger_test();

    void deleteFile(const std::string filePath);
};

}

#endif // LOGGER_TEST_H
