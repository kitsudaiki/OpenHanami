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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/logger.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <deque>
#include <mutex>
#include <sstream>

namespace Hanami
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

    bool writeDataIntoFile(const void *data,
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

}  // namespace Hanami

#endif  // BINARY_FILE_H
