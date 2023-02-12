/**
 *  @file    text_file_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TEXT_FILE_TEST_H
#define TEXT_FILE_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>
#include <filesystem>

namespace Kitsunemimi
{

class TextFile_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    TextFile_Test();

private:
    void initTest();
    void writeFile_test();
    void readFile_test();
    void appendText_test();
    void replaceLine_test();
    void replaceContent_test();
    void closeTest();

    std::string m_filePath = "";
    void deleteFile();
};

}

#endif // TEXT_FILE_TEST_H
