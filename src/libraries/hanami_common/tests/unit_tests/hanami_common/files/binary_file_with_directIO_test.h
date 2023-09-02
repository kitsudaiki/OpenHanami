/**
 *  @file    binary_file_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef BINARY_FILE_WITH_DIRECTIO_TEST_H
#define BINARY_FILE_WITH_DIRECTIO_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>
#include <filesystem>

namespace Hanami
{

class BinaryFile_withDirectIO_Test
        : public Hanami::CompareTestHelper
{
public:
    BinaryFile_withDirectIO_Test();

private:
    void initTest();
    void closeFile_test();
    void updateFileSize_test();
    void allocateStorage_test();
    void writeSegment_test();
    void readSegment_test();
    void writeCompleteFile_test();
    void readCompleteFile_test();
    void closeTest();

    std::string m_filePath = "";
    void deleteFile();
};

}

#endif // BINARY_FILE_WITH_DIRECTIO_TEST_H
