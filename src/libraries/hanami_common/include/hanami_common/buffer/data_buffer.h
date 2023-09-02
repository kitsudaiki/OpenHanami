/**
 *  @file       data_buffer.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#ifndef DATA_BUFFER_H
#define DATA_BUFFER_H

#include <string.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <hanami_common/memory_counter.h>

namespace Hanami
{

struct DataBuffer;
inline bool allocateBlocks_DataBuffer(DataBuffer &buffer, const uint64_t numberOfBlocks);
inline bool addData_DataBuffer(DataBuffer &buffer, const void* data, const uint64_t dataSize);
inline bool reset_DataBuffer(DataBuffer &buffer, const uint64_t numberOfBlocks);
inline uint8_t* getBlock_DataBuffer(DataBuffer &buffer, const uint64_t blockPosition);
inline void* alignedMalloc(const uint16_t blockSize, const uint64_t numberOfBytes);
inline bool alignedFree(void* ptr, const uint64_t numberOfBytes);

/**
 * @brief calculate number of block for a specific number of bytes
 *
 * @param numberOfBytes incoming number of bytes
 * @param blockSize bytes per block
 *
 * @return number of blocks
 */
inline uint64_t
calcBytesToBlocks(const uint64_t numberOfBytes,
                  const uint16_t blockSize = 4096)
{
    if(numberOfBytes % blockSize == 0) {
        return (numberOfBytes / blockSize);
    }

    return (numberOfBytes / blockSize) + 1;
}

struct DataBuffer
{
    uint64_t numberOfBlocks = 0;
    uint64_t usedBufferSize = 0;
    uint64_t totalBufferSize = 0;
    void* data = nullptr;
    uint16_t blockSize = 4096;
    uint8_t inUse = 0;
    // padding to expand the size of the struct to a multiple of 8
    uint8_t padding[5];

    /**
     * @brief constructor
     *
     * @param numberOfBlocks number of block of the initial allocation
     *                       (at least one)
     * @param blockSize size of a block in the data-buffer
     *                  (should never be changed after buffer was created)
     */
    DataBuffer(const uint64_t numberOfBlocks = 1,
               const uint16_t blockSize = 4096)
    {
        this->blockSize = blockSize;
        assert(this->blockSize % 8 == 0);
        if(numberOfBlocks < 1) {
            allocateBlocks_DataBuffer(*this, 1);
        }
        allocateBlocks_DataBuffer(*this, numberOfBlocks);
    }

    /**
     * @brief copy-constructor
     */
    DataBuffer(const DataBuffer &other)
    {
        // copy blockSize first to make sure, that the reset reallocate the correct total memroy
        this->blockSize = other.blockSize;
        allocateBlocks_DataBuffer(*this, other.numberOfBlocks);
        assert(this->totalBufferSize == other.totalBufferSize);
        assert(this->numberOfBlocks == other.numberOfBlocks);

        this->inUse = other.inUse;
        this->usedBufferSize = other.usedBufferSize;
        memcpy(data, other.data, this->usedBufferSize);
    }

    /**
     * @brief Simple additonal construct which use allready allocated memory.
     *        If this existing buffer is not a multiple of the blocksize,
     *        it allocate new memory with a valid size.
     *
     * @param data pointer to the already allocated memory
     * @param size size of the allocated memory
     */
    DataBuffer(void* data, const uint64_t size)
    {
        if(data == nullptr
                && size > 0)
        {
            this->data = data;
            numberOfBlocks =  calcBytesToBlocks(size);
            totalBufferSize = blockSize * numberOfBlocks;

            if(size % blockSize != 0) {
                allocateBlocks_DataBuffer(*this, 1);
            }
        }
    }

    /**
     * @brief copy-assignment operator
     */
    DataBuffer &operator=(const DataBuffer &other)
    {
        if(this != &other)
        {
            clear();

            // copy blockSize first to make sure, that the reset reallocate the correct total memroy
            this->blockSize = other.blockSize;
            allocateBlocks_DataBuffer(*this, other.numberOfBlocks);
            assert(this->totalBufferSize == other.totalBufferSize);
            assert(this->numberOfBlocks == other.numberOfBlocks);

            this->inUse = other.inUse;
            this->usedBufferSize = other.usedBufferSize;
            memcpy(data, other.data, this->usedBufferSize);
        }

        return *this;
    }

    /**
     * @brief destructor to clear the allocated memory inside this object
     */
    ~DataBuffer()
    {
        clear();
    }

    /**
     * @brief clear data of buffer
     *
     * @return false if buffer is not initialized and used, else true
     */
    bool clear()
    {
        // deallocate the buffer
        if(data != nullptr
                && inUse == 1)
        {
            alignedFree(data, totalBufferSize);
            inUse = 0;
            data = nullptr;
            numberOfBlocks = 0;

            return true;
        }

        return false;
    }
};

/**
 * @brief allocate a number of aligned bytes
 *
 * @param blockSize size of a single block for alignment. MUST be a multiple of 512.
 * @param numberOfBytes bytes to allocate
 *
 * @return pointer to the allocated memory or nullptr if blocksize is not a multiple of 512
 *         or allocation failed
 */
inline void*
alignedMalloc(const uint16_t blockSize,
              const uint64_t numberOfBytes)
{
    // precheck
    if(blockSize % 8 != 0) {
        return nullptr;
    }

    // allocate new memory
    void* ptr = nullptr;
    const int ret = posix_memalign(&ptr, blockSize, numberOfBytes);
    if(ret != 0) {
        return nullptr;
    }

    // update memory-counter in case of memory-leak-tests
    Hanami::increaseGlobalMemoryCounter(numberOfBytes);

    // init memory
    memset(ptr, 0, numberOfBytes);

    return ptr;
}

/**
 * @brief free aligned memory
 *        this method is a bit useless, but I wanted a equivalent for the alignedMalloc-method
 *
 * @param  ptr pointer to the memory to free
 * @param numberOfBytes bytes to free (only used for the memory-counter)
 *
 * @return true, if pointer not nullptr, else false
 */
inline bool
alignedFree(void* ptr,
            const uint64_t numberOfBytes)
{
    // precheck
    if(ptr == nullptr) {
        return false;
    }

    // update memory-counter in case of memory-leak-tests
    Hanami::decreaseGlobalMemoryCounter(numberOfBytes);

    // free data
    free(ptr);

    return true;
}

/**
 * @brief allocate more memory for the buffer.
 *        It allocates a bigger memory-block an copy the old buffer-content into the new.
 *
 * @param buffer reference to buffer-object
 * @param numberOfBlocks number of blocks to allocate
 *
 * @return true, if successful, else false
 */
inline bool
allocateBlocks_DataBuffer(DataBuffer &buffer,
                          const uint64_t numberOfBlocks)
{
    // create the new buffer
    const uint64_t newNumberOfBlocks = numberOfBlocks + buffer.numberOfBlocks;
    void* newBuffer =  alignedMalloc(buffer.blockSize, newNumberOfBlocks * buffer.blockSize);
    if(newBuffer == nullptr) {
        return false;
    }

    // copy the content of the old buffer to the new and deallocate the old
    if(buffer.data != nullptr
            && buffer.inUse == 1)
    {
        memcpy(newBuffer, buffer.data, buffer.numberOfBlocks * buffer.blockSize);
        alignedFree(buffer.data, buffer.totalBufferSize);
    }

    // set the new values
    buffer.inUse = 1;
    buffer.numberOfBlocks = newNumberOfBlocks;
    buffer.totalBufferSize = newNumberOfBlocks * buffer.blockSize;
    buffer.data = newBuffer;

    return true;
}

/**
 * @brief copy data into the buffer and resize the buffer in necessary
 *
 * @param buffer reference to buffer-object
 * @param data pointer the the data, which should be written into the buffer
 * @param dataSize number of bytes to write
 *
 * @return false if precheck or allocation failed, else true
 */
inline bool
addData_DataBuffer(DataBuffer &buffer,
                   const void* data,
                   const uint64_t dataSize)
{
    // precheck
    if(dataSize == 0
            || data == nullptr)
    {
        return false;
    }

    // check buffer-size and allocate more memory if necessary
    if(buffer.usedBufferSize + dataSize >= buffer.numberOfBlocks * buffer.blockSize)
    {
        const uint64_t newBlockNum = (dataSize / buffer.blockSize) + 1;
        if(allocateBlocks_DataBuffer(buffer, newBlockNum) == false) {
            return false;
        }
    }

    // copy the new data into the buffer
    uint8_t* dataByte = static_cast<uint8_t*>(buffer.data);
    memcpy(&dataByte[buffer.usedBufferSize], data, dataSize);
    buffer.usedBufferSize += dataSize;

    return true;
}

/**
 * @brief add an object to the buffer
 *
 * @param buffer reference to buffer-object
 * @param data pointer to the object, which shoulb be written to the buffer
 *
 * @return false if precheck or allocation failed, else true
 */
template <typename T>
inline bool
addObject_DataBuffer(DataBuffer &buffer, T* data)
{
    return addData_DataBuffer(buffer, data, sizeof(T));
}

/**
 * @brief reset a buffer and clears the data, so it is like the buffer is totally new
 *
 * @param buffer reference to buffer-object
 * @param numberOfBlocks number of new allocated blocks after buffer-reset
 *
 * @return false if precheck or allocation failed, else true
 */
inline bool
reset_DataBuffer(DataBuffer &buffer,
                 const uint64_t numberOfBlocks)
{
    // precheck
    if(numberOfBlocks == 0) {
        return false;
    }

    // deallocate ald buffer if possible
    if(buffer.data != nullptr
            && buffer.inUse == 1)
    {
        alignedFree(buffer.data, buffer.totalBufferSize);
    }

    // allocate at least one single block as new buffer-data
    void* newBuffer = alignedMalloc(buffer.blockSize, numberOfBlocks * buffer.blockSize);
    if(newBuffer == nullptr) {
        return false;
    }

    // reset metadata of the buffer
    buffer.data = newBuffer;
    buffer.inUse = 1;
    buffer.usedBufferSize = 0;
    buffer.totalBufferSize = numberOfBlocks * buffer.blockSize;
    buffer.numberOfBlocks = numberOfBlocks;

    return true;
}

/**
 * @brief get a pointer to a specific block inside the buffer
 *
 * @param buffer reference to buffer-object
 * @param blockPosition number of the block inside the buffer
 *
 * @return pointer to the buffer-position
 */
inline uint8_t*
getBlock_DataBuffer(DataBuffer &buffer,
                    const uint64_t blockPosition)
{
    // precheck
    if(blockPosition >= buffer.numberOfBlocks) {
        return nullptr;
    }

    // get specific block of the data
    uint8_t* dataByte = static_cast<uint8_t*>(buffer.data);
    return &dataByte[blockPosition * buffer.blockSize];
}

}

#endif // DATABUFFER_H
