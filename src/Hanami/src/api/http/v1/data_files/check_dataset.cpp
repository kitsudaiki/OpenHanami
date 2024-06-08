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

#include <core/io/data_set/dataset_file_io.h>
#include <database/dataset_table.h>
#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

CheckMnistDataSet::CheckMnistDataSet()
    : Blossom("Compare a list of values with a dataset to check accuracy.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("result_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset to compare to.")
        .setRegex(UUID_REGEX);

    registerInputField("dataset_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset with the results, which should be checked.")
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
CheckMnistDataSet::runTask(BlossomIO& blossomIO,
                           const json& context,
                           BlossomStatus& status,
                           Hanami::ErrorContainer& error)
{
    const std::string resultUuid = blossomIO.input["result_uuid"];
    const std::string datasetUuid = blossomIO.input["dataset_uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    DataSetFileHandle datasetFileHandle;
    DataSetFileHandle resultFileHandle;

    // open files
    ReturnStatus ret = getFileHandle(datasetFileHandle, datasetUuid, userContext, status, error);
    if (ret != OK) {
        return false;
    }
    ret = getFileHandle(resultFileHandle, resultUuid, userContext, status, error);
    if (ret != OK) {
        return false;
    }

    // set file-selectors
    resultFileHandle.readSelector.endColumn = 10;
    resultFileHandle.readSelector.endRow = 10000;
    datasetFileHandle.readSelector.startColumn = 784;
    datasetFileHandle.readSelector.endColumn = 794;
    datasetFileHandle.readSelector.endRow = 10000;

    // init buffer for output
    std::vector<float> datasetOutput(10, 0.0f);
    std::vector<float> resultOutput(10, 0.0f);

    float accuracy = 0.0f;

    // check files
    for (uint64_t row = 0; row < 10000; row++) {
        if (getDataFromDataSet(datasetOutput, datasetFileHandle, row, error) != OK) {
            status.statusCode = INVALID_INPUT;
            status.errorMessage
                = "Dataset with UUID '" + datasetUuid + "' is invalid and can not be compared";
            error.addMessage(status.errorMessage);
            return false;
        }
        if (getDataFromDataSet(resultOutput, resultFileHandle, row, error) != OK) {
            status.statusCode = INVALID_INPUT;
            status.errorMessage = "Dataset with result with UUID '" + resultUuid
                                  + "' is invalid and can not be checked";
            error.addMessage(status.errorMessage);
            return false;
        }

        bool allCorrect = true;
        for (uint64_t i = 0; i < 10; i++) {
            if (datasetOutput[i] != resultOutput[i]) {
                allCorrect = false;
            }
        }

        if (allCorrect) {
            accuracy += 1.0f;
        }
    }

    blossomIO.output["accuracy"] = (100.0f / 10000.0f) * accuracy;

    return true;
}

/**
 * @brief CheckMnistDataSet::getFileHandle
 * @param fileHandle
 * @param uuid
 * @param context
 * @param status
 * @param error
 * @return
 */
ReturnStatus
CheckMnistDataSet::getFileHandle(DataSetFileHandle& fileHandle,
                                 const std::string uuid,
                                 const Hanami::UserContext userContext,
                                 BlossomStatus& status,
                                 Hanami::ErrorContainer& error)
{
    json dbOutput;
    ReturnStatus ret
        = DataSetTable::getInstance()->getDataSet(dbOutput, uuid, userContext, true, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return ret;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Dataset with uuid '" + uuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return ret;
    }

    // get file information
    const std::string location = dbOutput["location"];
    if (openDataSetFile(fileHandle, location, error) != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return ERROR;
    }

    return OK;
}
