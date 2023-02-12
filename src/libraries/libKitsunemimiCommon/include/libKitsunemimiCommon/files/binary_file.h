/**
 *  @file    binary_file.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 *
 *  @brief class for binary-file-handling
 */

#ifndef BINARY_FILE_H
#define BINARY_FILE_H

#include <deque>
#include <sstream>
#include <mutex>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/buffer/data_buffer.h>

namespace Kitsunemimi
{

class BinaryFile
{
public:
    BinaryFile(const std::string &filePath);
    ~BinaryFile();

    bool allocateStorage(const uint64_t numberOfBytes, ErrorContainer &error);
    bool updateFileSize(ErrorContainer &error);

    bool readCompleteFile(DataBuffer &buffer, ErrorContainer &error);
    bool writeCompleteFile(DataBuffer &buffer, ErrorContainer &error);

    bool writeDataIntoFile(const void* data,
                           const uint64_t startBytePosition,
                           const uint64_t numberOfBytes,
                           ErrorContainer &error);
    bool readDataFromFile(void *data,
                          const uint64_t startBytePosition,
                          const uint64_t numberOfBytes,
                          ErrorContainer &error);

    bool closeFile(ErrorContainer &error);

    // public variables to avoid stupid getter
    uint64_t m_totalFileSize = 0;
    std::string m_filePath = "";

private:
    int m_fileDescriptor = -1;

    bool initFile(ErrorContainer &error);
};

}

#endif // BINARY_FILE_H
