/**
 *  @file    binary_file_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef BINARY_FILE_WITHOUT_DIRECTIO_TEST_H
#define BINARY_FILE_WITHOUT_DIRECTIO_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>
#include <filesystem>

namespace Kitsunemimi
{

class BinaryFile_withoutDirectIO_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    BinaryFile_withoutDirectIO_Test();

private:
    void initTest();
    void closeFile_test();
    void updateFileSize_test();
    void allocateStorage_test();
    void writeCompleteFile_test();
    void readCompleteFile_test();
    void writeDataIntoFile_test();
    void readDataFromFile_test();
    void closeTest();

    std::string m_filePath = "";
    void deleteFile();
};

} // namespace Kitsunemimi

#endif // BINARY_FILE_WITHOUT_DIRECTIO_TEST_H
