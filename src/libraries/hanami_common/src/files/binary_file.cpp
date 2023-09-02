/**
 *  @file    binary_file.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 *
 *  @brief class for binary-file-handling
 */

#include <hanami_common/files/binary_file.h>

using Kitsunemimi::DataBuffer;

namespace Kitsunemimi
{

/**
 * @brief constructor
 *
 * @param filePath file-path of the binary-file
 */
BinaryFile::BinaryFile(const std::string &filePath)
{
    m_filePath = filePath;

    ErrorContainer error;
    if(initFile(error) == false) {
        LOG_ERROR(error);
    }
}

/**
 * @brief destructor
 */
BinaryFile::~BinaryFile()
{
    ErrorContainer error;
    if(closeFile(error) == false) {
        LOG_ERROR(error);
    }
}

/**
 * @brief create a new file or open an existing file
 *
 * @param error reference for error-output
 *
 * @return true is successful, else false
 */
bool
BinaryFile::initFile(Kitsunemimi::ErrorContainer &error)
{
    m_fileDescriptor = open(m_filePath.c_str(),
                            O_CREAT | O_RDWR | O_LARGEFILE,
                            0666);

    // check if file is open
    if(m_fileDescriptor == -1)
    {
        error.addMeesage("Failed to initialize binary file for path '" + m_filePath + "'");
        return false;
    }

    return updateFileSize(error);
}

/**
 * @brief allocate new storage at the end of the file
 *
 * @param numberOfBytes number of bytes to allocate additionally to allready allocated
 * @param error reference for error-output
 *
 * @return true is successful, else false
 */
bool
BinaryFile::allocateStorage(const uint64_t numberOfBytes, ErrorContainer &error)
{
    if(numberOfBytes == 0) {
        return true;
    }

    // set first to the start of the file and allocate the new size at the end of the file
    lseek(m_fileDescriptor, 0, SEEK_SET);
    const long ret = posix_fallocate(m_fileDescriptor,
                                     static_cast<long>(m_totalFileSize),
                                     static_cast<long>(numberOfBytes));

    // check if allocation was successful
    if(ret != 0)
    {
        // TODO: process errno
        error.addMeesage("Failed to allocate new storage for the binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    // got the the end of the file
    return updateFileSize(error);
}

/**
 * @brief update size-information from the file
 *
 * @param error reference for error-output
 *
 * @return false, if file not open, else true
 */
bool
BinaryFile::updateFileSize(ErrorContainer &error)
{
    if(m_fileDescriptor == -1)
    {
        error.addMeesage("Failed to allocate new storage for the binary file for path '"
                         + m_filePath
                         + "', because the file is not open.");
        return false;
    }

    // check if filesize is really 0 or check is requested
    const long ret = lseek(m_fileDescriptor, 0, SEEK_END);
    if(ret >= 0) {
        m_totalFileSize = static_cast<uint64_t>(ret);
    }

    lseek(m_fileDescriptor, 0, SEEK_SET);

    return true;
}

/**
 * @brief read a complete binary file into a data-buffer object
 *
 * @param buffer reference to the buffer, where the data should be written into
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
BinaryFile::readCompleteFile(DataBuffer &buffer, ErrorContainer &error)
{
    // go to the end of the file to get the size of the file
    const long size = lseek(m_fileDescriptor, 0, SEEK_END);
    if(size <= 0)
    {
        error.addMeesage("Failed to find the end of the binary file for path '"
                         + m_filePath
                         + "'.");
        return false;
    }

    // resize buffer to the size of the file
    uint64_t numberOfBlocks = (static_cast<uint64_t>(size) / buffer.blockSize);
    if(size % buffer.blockSize != 0) {
        numberOfBlocks++;
    }
    allocateBlocks_DataBuffer(buffer, numberOfBlocks);

    // go to the beginning of the file again and read the complete file into the buffer
    lseek(m_fileDescriptor, 0, SEEK_SET);
    const ssize_t ret = read(m_fileDescriptor, buffer.data, static_cast<uint64_t>(size));
    if(ret == -1)
    {
        // TODO: process errno
        error.addMeesage("Failed to read the binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    // size buffer-size
    buffer.usedBufferSize = static_cast<uint64_t>(size);

    return true;
}

/**
 * @brief write a complete buffer into a binary-file
 *
 * @param buffer reference to the buffer with the data, which should be written into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
BinaryFile::writeCompleteFile(DataBuffer &buffer, ErrorContainer &error)
{
    // resize file to the size of the buffer
    int64_t sizeDiff = buffer.usedBufferSize - m_totalFileSize;
    if(sizeDiff > 0)
    {
        // allocate additional memory
        if(allocateStorage(sizeDiff, error) == false)
        {
            error.addMeesage("Failed to write to binary file for path '"
                             + m_filePath
                             + "'");
            return false;
        }
    }

    // go to the beginning of the file and write data to file
    lseek(m_fileDescriptor, 0, SEEK_SET);
    const ssize_t ret = write(m_fileDescriptor, buffer.data, buffer.usedBufferSize);
    if(ret == -1)
    {
        // TODO: process errno
        error.addMeesage("Failed to write to binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    return true;
}

/**
 * @brief write data to a spicific position of the file, but only for for files
 *
 * @param data pointer to the buffer where the data coming from
 * @param startBytePosition position in file where to start to write
 * @param numberOfBytes number of bytes to write to file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
BinaryFile::writeDataIntoFile(const void* data,
                              const uint64_t startBytePosition,
                              const uint64_t numberOfBytes,
                              ErrorContainer &error)
{
    if(numberOfBytes == 0) {
        return true;
    }

    // precheck
    if(startBytePosition + numberOfBytes > m_totalFileSize
            || m_fileDescriptor < 0)
    {
        error.addMeesage("Failed to write data to binary file for path '"
                         + m_filePath
                         + "', because the precheck failed. Either the buffer is incompatible "
                           "or the file is not open.");
        return false;
    }

    // go to the requested position and write the block
    const long retSeek = lseek(m_fileDescriptor,
                               static_cast<long>(startBytePosition),
                               SEEK_SET);
    if(retSeek < 0)
    {
        error.addMeesage("Failed to go to the requested write position in binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    // write data to file
    const ssize_t ret = write(m_fileDescriptor, static_cast<const uint8_t*>(data), numberOfBytes);
    if(ret == -1)
    {
        // TODO: process errno
        error.addMeesage("Failed to write data to binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    // sync file
    fdatasync(m_fileDescriptor);

    return true;
}

/**
 * @brief read data from a spicific position of the file, but only for for files
 *
 * @param data pointer to the buffer where the data of the file should written into
 * @param startBytePosition position in file where to start to read
 * @param numberOfBytes number of bytes to read from file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
BinaryFile::readDataFromFile(void* data,
                             const uint64_t startBytePosition,
                             const uint64_t numberOfBytes,
                             ErrorContainer &error)
{
    if(numberOfBytes == 0) {
        return true;
    }

    // precheck
    if(startBytePosition + numberOfBytes > m_totalFileSize
            || m_fileDescriptor < 0)
    {
        error.addMeesage("Failed to read data of binary file for path '"
                         + m_filePath
                         + "', because the precheck failed. Either the buffer is incompatible "
                           "or the file is not open.");
        return false;
    }

    // go to the requested position and read the block
    const long retSeek = lseek(m_fileDescriptor,
                               static_cast<long>(startBytePosition),
                               SEEK_SET);
    if(retSeek < 0)
    {
        error.addMeesage("Failed to go to the requested read position in binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    const ssize_t ret = read(m_fileDescriptor, static_cast<uint8_t*>(data), numberOfBytes);
    if(ret == -1)
    {
        // TODO: process errno
        error.addMeesage("Failed to read data of binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    return true;
}

/**
 * @brief close the cluser-file
 *
 * @param error reference for error-output
 *
 * @return true, if file-descriptor is valid, else false
 */
bool
BinaryFile::closeFile(ErrorContainer &error)
{
    // precheck
    if(m_fileDescriptor == -1) {
        return true;
    }

    // try to close file
    if(close(m_fileDescriptor) < 0)
    {
        // TODO: process errno
        error.addMeesage("Failed to close binary file for path '"
                         + m_filePath
                         + "'");
        return false;
    }

    m_fileDescriptor = -1;
    return true;
}

}
