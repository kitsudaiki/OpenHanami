/**
 * @file        dataset_file_io.cpp
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

#ifndef HANAMI_DATASETFILE_H
#define HANAMI_DATASETFILE_H

#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/enums.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/logger.h>
#include <stdint.h>

#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

enum DataSetType : uint8_t {
    UNDEFINED_TYPE = 0,
    UNIN8_TYPE = 1,
    UNIN32_TYPE = 2,
    UNIN64_TYPE = 4,
    FLOAT_TYPE = 5,
};

struct DataSetSelector {
    uint64_t startRow = 0;
    uint64_t endRow = 0;
    uint64_t startColumn = 0;
    uint64_t endColumn = 0;
};

struct DataSetHeader {
    const char typeIdentifier[8] = "hanami";
    const char fileIdentifier[32] = "data-set";

    uint8_t version = 1;
    DataSetType dataType = UNDEFINED_TYPE;

    uint8_t padding1[6];

    uint64_t fileSize = 0;
    uint64_t numberOfRows = 0;
    uint64_t numberOfColumns = 0;

    char name[256];
    uint32_t nameSize = 0;

    uint8_t padding2[180];

    DataSetHeader() { memset(name, 0, 256); }

    bool setName(const std::string& newName)
    {
        // precheck
        if (newName.size() > 255 || newName.size() == 0) {
            return false;
        }

        // copy string into char-buffer and set explicit the escape symbol to be absolut sure
        // that it is set to absolut avoid buffer-overflows
        strncpy(name, newName.c_str(), newName.size());
        name[newName.size()] = '\0';
        nameSize = newName.size();

        return true;
    }

    const std::string getName() const
    {
        // precheck
        if (nameSize == 0 || nameSize > 255) {
            return std::string("");
        }

        return std::string(name, nameSize);
    }

    void toJson(json& result) const
    {
        result = json::object();
        result["version"] = version;
        result["name"] = getName();
        result["data_type"] = dataType;
        result["file_size"] = fileSize;
        result["number_of_rows"] = numberOfRows;
        result["number_of_columns"] = numberOfColumns;
    }
};
static_assert(sizeof(DataSetHeader) == 512);

struct DataSetFileHandle {
    DataSetHeader header;
    Hanami::BinaryFile* targetFile = nullptr;

    ~DataSetFileHandle()
    {
        if (targetFile != nullptr) {
            Hanami::ErrorContainer error;
            if (updateHeaderInFile(error) == false) {
                error.addMessage("Failed to update data-set header in file while closing");
                LOG_ERROR(error);
            }
            if (targetFile->closeFile(error) == false) {
                error.addMessage("Failed to close data-set-file");
                LOG_ERROR(error);
            }
        }
    }

    bool updateHeaderInFile(Hanami::ErrorContainer& error)
    {
        if (targetFile == nullptr) {
            return true;
        }
        if (targetFile->writeDataIntoFile(&header, 0, sizeof(DataSetHeader), error) == false) {
            error.addMessage("Failed to write data-set header to disc");
            return false;
        }
        return true;
    }
};

/**
 * @brief copy data from the temporary buffer into the final output
 *
 * @param result reference to the output-list for the final values
 * @param buffer temporary buffer with the data
 * @param selector selector with the range-values of the selected area
 * @param maxColumns maximum amount of columns in the data-set
 */
template <typename T>
void
copyDataSetDate(std::vector<float>& result,
                const Hanami::DataBuffer& buffer,
                const DataSetSelector selector,
                const uint64_t maxColumns)
{
    const T* data = static_cast<const T*>(buffer.data);
    uint64_t counter = 0;
    uint64_t currentPos = 0;
    for (uint64_t y = 0; y < (selector.endRow - selector.startRow); y++) {
        for (uint64_t x = selector.startColumn; x < selector.endColumn; x++) {
            currentPos = (y * maxColumns) + x;
            result[counter] = static_cast<float>(data[currentPos]);
            counter++;
        }
    }
}

/**
 * @brief append new row
 *
 * @param fileHandle handle of the data-set file, where the new data should be appended
 * @param input json-array as input with the new data to append.
 *              Must be a multiple of the number of columns in the data-set.
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
template <typename T>
ReturnStatus
appendToDataSet(DataSetFileHandle& fileHandle, const json& input, Hanami::ErrorContainer& error)
{
    if (input.is_array() == false || input.size() == 0) {
        return INVALID_INPUT;
    }

    if (input.size() % fileHandle.header.numberOfColumns != 0) {
        return INVALID_INPUT;
    }

    std::vector<T> temp;
    temp.resize(input.size());
    for (uint64_t i = 0; i < input.size(); i++) {
        if (input[i].is_number() == false) {
            return INVALID_INPUT;
        }

        temp[i] = input[i];
    }

    const uint64_t offset = fileHandle.targetFile->fileSize;
    const uint64_t additionalSize = temp.size() * sizeof(T);
    if (fileHandle.targetFile->allocateStorage(additionalSize, error) == false) {
        return ERROR;
    }

    if (fileHandle.targetFile->writeDataIntoFile(&temp[0], offset, additionalSize, error) == false)
    {
        return ERROR;
    }

    fileHandle.header.fileSize = fileHandle.targetFile->fileSize;
    fileHandle.header.numberOfRows += temp.size() / fileHandle.header.numberOfColumns;

    return OK;
}

ReturnStatus openDataSetFile(DataSetFileHandle& result,
                             const std::string& filePath,
                             Hanami::ErrorContainer& error);

ReturnStatus initNewDataSetFile(DataSetFileHandle& result,
                                const std::string& filePath,
                                const std::string& name,
                                const DataSetType type,
                                const uint64_t numberOfColumns,
                                Hanami::ErrorContainer& error);

ReturnStatus getDataFromDataSet(std::vector<float>& result,
                                const DataSetFileHandle& fileHandle,
                                const DataSetSelector selector,
                                Hanami::ErrorContainer& error);

#endif  // HANAMI_DATASETFILE_H
