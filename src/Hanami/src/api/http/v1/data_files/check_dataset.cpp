/**
 * @file        check_dataset.cpp
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

#include "check_dataset.h"

#include <database/dataset_table.h>
#include <database/request_result_table.h>
#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_config/config_handler.h>
#include <hanami_files/dataset_files/dataset_file.h>
#include <hanami_files/dataset_files/image_dataset_file.h>
#include <hanami_root.h>

CheckDataSet::CheckDataSet() : Blossom("Compare a list of values with a dataset to check accuracy.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("result_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset to compare to.")
        .setRegex(UUID_REGEX);

    registerInputField("dataset_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset to compare to.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("accuracy", SAKURA_FLOAT_TYPE)
        .setComment("Correctness of the values compared to the dataset.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CheckDataSet::runTask(BlossomIO& blossomIO,
                      const json& context,
                      BlossomStatus& status,
                      Hanami::ErrorContainer& error)
{
    const std::string resultUuid = blossomIO.input["result_uuid"];
    const std::string dataUuid = blossomIO.input["dataset_uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    // get result
    // check if request-result exist within the table
    json requestResult;
    if (RequestResultTable::getInstance()->getRequestResult(
            requestResult, resultUuid, userContext, error, true)
        == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (requestResult.size() == 0) {
        status.errorMessage = "Result with uuid '" + resultUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get data-info from database
    json dbOutput;
    if (DataSetTable::getInstance()->getDataSet(dbOutput, dataUuid, userContext, error, true)
        == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get file information
    const std::string location = dbOutput["location"];

    Hanami::DataBuffer buffer;
    DataSetFile::DataSetHeader dataSetHeader;
    ImageDataSetFile::ImageTypeHeader imageTypeHeader;
    Hanami::BinaryFile file(location);

    // read dataset-header
    if (file.readCompleteFile(buffer, error) == false) {
        error.addMessage("Failed to read dataset-header from file '" + location + "'");
        return false;
    }

    // prepare values
    uint64_t correctValues = 0;
    uint64_t dataPos
        = sizeof(DataSetFile::DataSetHeader) + sizeof(ImageDataSetFile::ImageTypeHeader);
    const uint8_t* u8Data = static_cast<const uint8_t*>(buffer.data);
    memcpy(&dataSetHeader, buffer.data, sizeof(DataSetFile::DataSetHeader));
    memcpy(&imageTypeHeader,
           &u8Data[sizeof(DataSetFile::DataSetHeader)],
           sizeof(ImageDataSetFile::ImageTypeHeader));
    const uint64_t lineOffset = imageTypeHeader.numberOfInputsX * imageTypeHeader.numberOfInputsY;
    const uint64_t lineSize = (imageTypeHeader.numberOfInputsX * imageTypeHeader.numberOfInputsY)
                              + imageTypeHeader.numberOfOutputs;
    const float* content = reinterpret_cast<const float*>(&u8Data[dataPos]);

    // iterate over all values and check
    json compareData = requestResult["data"];
    for (uint64_t i = 0; i < compareData.size(); i++) {
        const uint64_t actualPos = (i * lineSize) + lineOffset;
        const uint64_t checkVal = compareData[i];
        if (content[actualPos + checkVal] > 0.0f) {
            correctValues++;
        }
    }

    // add result to output
    const float accuracy
        = (100.0f / static_cast<float>(compareData.size())) * static_cast<float>(correctValues);
    blossomIO.output["accuracy"] = accuracy;

    return true;
}
