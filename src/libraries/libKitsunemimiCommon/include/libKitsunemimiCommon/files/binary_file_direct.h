/**
 *  @file    binary_file_direct.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 *
 *  @brief class for binary-file-handling
 */

#ifndef BINARY_FILE_DIRECT_H
#define BINARY_FILE_DIRECT_H

#include <deque>
#include <sstream>
#include <mutex>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

class BinaryFileDirect
{
public:
    BinaryFileDirect(const std::string &filePath);
    ~BinaryFileDirect();

    bool allocateStorage(const uint64_t numberOfBlocks,
                         const uint32_t blockSize,
                         ErrorContainer &error);
    bool updateFileSize(ErrorContainer &error);

    bool readCompleteFile(DataBuffer &buffer, ErrorContainer &error);
    bool writeCompleteFile(DataBuffer &buffer, ErrorContainer &error);

    bool readSegment(DataBuffer &buffer,
                     const uint64_t startBlockInFile,
                     const uint64_t numberOfBlocks,
                     const uint64_t startBlockInBuffer,
                     ErrorContainer &error);
    bool writeSegment(DataBuffer &buffer,
                      const uint64_t startBlockInFile,
                      const uint64_t numberOfBlocks,
                      const uint64_t startBlockInBuffer,
                      ErrorContainer &error);

    bool closeFile(ErrorContainer &error);

    // public variables to avoid stupid getter
    uint64_t m_totalFileSize = 0;
    std::string m_filePath = "";

private:
    int m_fileDescriptor = -1;
    uint16_t m_blockSize = 512;

    bool initFile(ErrorContainer &error);
    bool allocateStorage(const uint64_t numberOfBytes, ErrorContainer &error);
};

}

#endif // BINARY_FILE_DIRECT_H
