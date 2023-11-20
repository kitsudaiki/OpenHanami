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

#include <common/defines.h>
#include <common/enums.h>
#include <hanami_common/buffer/bit_buffer.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_policies/policy.h>
#include <stdint.h>
#include <uuid/uuid.h>

#include <string>

struct NextSides {
    uint8_t sides[5];
};

struct RequestMessage {
    Hanami::HttpRequestType httpType = Hanami::GET_TYPE;
    std::string id = "";
    std::string inputValues = "{}";
};

struct UserContext {
    std::string userId = "";
    std::string projectId = "";
    bool isAdmin = false;
    bool isProjectAdmin = false;
    std::string token = "";

    UserContext() {}

    UserContext(const json& inputContext)
    {
        userId = inputContext["id"];
        projectId = inputContext["project_id"];
        isAdmin = inputContext["is_admin"];
        isProjectAdmin = inputContext["is_project_admin"];
        if (inputContext.contains("token")) {
            token = inputContext["token"];
        }
    }
};

struct EndpointEntry {
    SakuraObjectType type = BLOSSOM_TYPE;
    std::string group = "-";
    std::string name = "";
};

struct BlossomStatus {
    uint64_t statusCode = OK_RTYPE;
    std::string errorMessage = "";
};

struct FileHandle {
    UserContext userContext;
    Hanami::BinaryFile* file = nullptr;
    Hanami::BitBuffer* bitBuffer = nullptr;

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

#endif  // HANAMI_STRUCTS_H
