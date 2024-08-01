/**
 * @file        finalize_mnist_dataset_v1_0.cpp
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

#include "finalize_mnist_dataset_v1_0.h"

#include <core/io/data_set/dataset_file_io.h>
#include <core/temp_file_handler.h>
#include <database/dataset_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_crypto/common.h>
#include <hanami_root.h>

FinalizeMnistDataSetV1M0::FinalizeMnistDataSetV1M0()
    : Blossom(
        "Finalize uploaded dataset by checking completeness of the "
        "uploaded and convert into generic format.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the new dataset.")
        .setRegex(UUID_REGEX);

    registerInputField("uuid_input_file", SAKURA_STRING_TYPE)
        .setComment("UUID to identify the file for date upload of input-data.")
        .setRegex(UUID_REGEX);

    registerInputField("uuid_label_file", SAKURA_STRING_TYPE)
        .setComment("UUID to identify the file for date upload of label-data.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the new dataset.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
FinalizeMnistDataSetV1M0::runTask(BlossomIO& blossomIO,
                                  const json& context,
                                  BlossomStatus& status,
                                  Hanami::ErrorContainer& error)
{
    const std::string uuid = blossomIO.input["uuid"];
    const std::string inputUuid = blossomIO.input["uuid_input_file"];
    const std::string labelUuid = blossomIO.input["uuid_label_file"];
    const Hanami::UserContext userContext = convertContext(context);

    // get location from database
    DataSetTable::DataSetDbEntry result;
    const ReturnStatus ret
        = DataSetTable::getInstance()->getDataSet(result, uuid, userContext, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + uuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // read input-data from temp-file
    Hanami::DataBuffer inputBuffer;
    if (TempFileHandler::getInstance()->getData(inputBuffer, inputUuid) == false) {
        status.errorMessage = "Input-data with uuid '" + inputUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // read label from temp-file
    Hanami::DataBuffer labelBuffer;
    if (TempFileHandler::getInstance()->getData(labelBuffer, labelUuid) == false) {
        status.errorMessage = "Label-data with uuid '" + labelUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // write data to file
    if (convertMnistData(result.location, result.name, inputBuffer, labelBuffer, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMessage("Failed to convert mnist-data");
        return false;
    }

    // delete temp-files
    // TODO: error-handling
    TempFileHandler::getInstance()->removeData(inputUuid, userContext, error);
    TempFileHandler::getInstance()->removeData(labelUuid, userContext, error);

    // create output
    blossomIO.output["uuid"] = uuid;

    return true;
}

/**
 * @brief convert mnist-data into generic format
 *
 * @param filePath path to the resulting file
 * @param name dataset name
 * @param inputBuffer buffer with input-data
 * @param labelBuffer buffer with label-data
 * @param error reference for error-output
 *
 * @return true, if successfull, else false
 */
bool
FinalizeMnistDataSetV1M0::convertMnistData(const std::string& filePath,
                                           const std::string& name,
                                           const Hanami::DataBuffer& inputBuffer,
                                           const Hanami::DataBuffer& labelBuffer,
                                           Hanami::ErrorContainer& error)
{
    // source-data
    const uint64_t dataOffset = 16;
    const uint64_t labelOffset = 8;
    const uint8_t* dataBufferPtr = static_cast<uint8_t*>(inputBuffer.data);
    const uint8_t* labelBufferPtr = static_cast<uint8_t*>(labelBuffer.data);

    // get number of images
    uint32_t numberOfImages = 0;
    numberOfImages |= dataBufferPtr[7];
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[6]) << 8;
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[5]) << 16;
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[4]) << 24;

    // get number of rows
    uint32_t numberOfRows = 0;
    numberOfRows |= dataBufferPtr[11];
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[10]) << 8;
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[9]) << 16;
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[8]) << 24;

    // get number of columns
    uint32_t numberOfColumns = 0;
    numberOfColumns |= dataBufferPtr[15];
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[14]) << 8;
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[13]) << 16;
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[12]) << 24;

    const uint32_t pictureSize = numberOfRows * numberOfColumns;
    const uint64_t lineSize = pictureSize + 10;

    // create description
    json description;
    json inputDescrEntry;
    inputDescrEntry["start_column"] = 0;
    inputDescrEntry["end_column"] = pictureSize;
    description["picture"] = inputDescrEntry;
    json outputDescrEntry;
    outputDescrEntry["start_column"] = pictureSize;
    outputDescrEntry["end_column"] = pictureSize + 10;
    description["label"] = outputDescrEntry;

    // initialize file
    DataSetFileHandle fileHandle;
    if (initNewDataSetFile(
            fileHandle, filePath, name, description, DataSetType::UINT8_TYPE, lineSize, error)
        != OK)
    {
        return false;
    }
    fileHandle.initReadWriteBuffer(10);

    // buffer for values
    std::vector<uint8_t> line(lineSize, 0.0f);

    // copy values of each pixel into the resulting file
    for (uint32_t pic = 0; pic < numberOfImages; pic++) {
        // input
        for (uint32_t i = 0; i < pictureSize; i++) {
            const uint32_t pos = pic * pictureSize + i + dataOffset;
            line[i] = dataBufferPtr[pos];
        }

        // label
        for (uint32_t i = 0; i < 10; i++) {
            line[pictureSize + i] = 0;
        }

        const uint32_t label = labelBufferPtr[pic + labelOffset];
        line[pictureSize + label] = 1;

        if (appendToDataSet(fileHandle, &line[0], line.size(), error) != OK) {
            return false;
        }
    }

    return true;
}
