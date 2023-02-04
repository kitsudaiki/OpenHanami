/**
 *  @file      file_methods_test.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef FILE_METHODS_TEST_H
#define FILE_METHODS_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class FileMethods_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    FileMethods_Test();

private:
    void listFiles_test();
    void renameFileOrDir_test();
    void copyPath_test();
    void createDirectory_test();
    void deleteFileOrDir_test();
};

}

#endif // FILE_METHODS_TEST_H
