/**
 * @file        structs.h
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#ifndef HANAMI_STRUCTS_H
#define HANAMI_STRUCTS_H

#include <stdint.h>

#include <string>

#include "buffer/bit_buffer.h"
#include "buffer/data_buffer.h"
#include "enums.h"
#include "files/binary_file.h"
#include "logger.h"

#define UNINTI_POINT_32 0x0FFFFFFF
#define TRANSFER_SEGMENT_SIZE (128 * 1024)

namespace Hanami
{

struct RequestMessage {
    HttpRequestType httpType = GET_TYPE;
    std::string id = "";
    std::string inputValues = "{}";
};

struct UserContext {
    std::string userId = "";
    std::string projectId = "";
    bool isAdmin = false;
    bool isProjectAdmin = false;
    std::string token = "";
};

struct Position {
    uint32_t x = UNINTI_POINT_32;
    uint32_t y = UNINTI_POINT_32;
    uint32_t z = UNINTI_POINT_32;
    uint32_t w = UNINTI_POINT_32;

    Position() {}

    Position(const Position& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    Position& operator=(const Position& other)
    {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        return *this;
    }

    bool operator==(const Position& other) const
    {
        return (this->x == other.x && this->y == other.y && this->z == other.z);
    }

    bool operator!=(const Position& other) const
    {
        return (this->x != other.x || this->y != other.y || this->z != other.z);
    }

    bool isValid() const
    {
        return (x != UNINTI_POINT_32 && y != UNINTI_POINT_32 && z != UNINTI_POINT_32);
    }

    const std::string toString() const
    {
        return "[ " + std::to_string(x) + " , " + std::to_string(y) + " , " + std::to_string(z)
               + " ]";
    }
};

struct FileHandle {
    Hanami::UserContext userContext;
    Hanami::BinaryFile* file = nullptr;
    Hanami::BitBuffer* bitBuffer = nullptr;
    uint64_t timeoutCounter = 0;
    bool lock = false;

    FileHandle() {}

    FileHandle(const FileHandle& other)
    {
        userContext = other.userContext;
        file = other.file;
        bitBuffer = other.bitBuffer;
    }

    ~FileHandle()
    {
        if (file != nullptr) {
            delete file;
        }
        if (bitBuffer != nullptr) {
            delete bitBuffer;
        }
    }

    FileHandle& operator=(const FileHandle& other)
    {
        userContext = other.userContext;
        file = other.file;
        bitBuffer = other.bitBuffer;
        return *this;
    }

    /**
     * @brief add data to the buffer of the file-handle
     *
     * @param pos byte-postion where to add data to the buffer
     * @param data data to write
     * @param size number of byte to write
     * @param errorMessage reference for error-output
     *
     * @return true, if successful, else false
     */
    bool addDataToPos(const uint64_t pos,
                      const void* data,
                      const uint64_t size,
                      std::string& errorMessage)
    {
        Hanami::ErrorContainer error;

        // write data to file
        if (file->writeDataIntoFile(data, pos, size, error) == false) {
            LOG_ERROR(error);
            errorMessage = "Failed to write data to disc";
            return false;
        }

        // update progress in bit-buffer
        bitBuffer->set(pos / TRANSFER_SEGMENT_SIZE, true);

        return true;
    }
};

}  // namespace Hanami

#endif  // HANAMI_STRUCTS_H
