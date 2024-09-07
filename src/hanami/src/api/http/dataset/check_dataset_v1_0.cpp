/**
 * @file        check_dataset_v1_0.cpp
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

#include "check_dataset_v1_0.h"

#include <core/io/data_set/dataset_file_io.h>
#include <database/dataset_table.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

CheckMnistDataSetV1M0::CheckMnistDataSetV1M0()
    : Blossom("Compare a list of values with a dataset to check accuracy.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("reference_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset to compare to.")
        .setRegex(UUID_REGEX);

    registerInputField("uuid", SAKURA_STRING_TYPE)
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
CheckMnistDataSetV1M0::runTask(BlossomIO& blossomIO,
                               const Hanami::UserContext& userContext,
                               BlossomStatus& status,
                               Hanami::ErrorContainer& error)
{
    const std::string referenceUuid = blossomIO.input["reference_uuid"];
    const std::string datasetUuid = blossomIO.input["uuid"];

    DataSetFileHandle referenceFileHandle;
    DataSetFileHandle datasetFileHandle;

    // open files
    ReturnStatus ret
        = getFileHandle(referenceFileHandle, referenceUuid, userContext, status, error);
    if (ret != OK) {
        return false;
    }
    ret = getFileHandle(datasetFileHandle, datasetUuid, userContext, status, error);
    if (ret != OK) {
        return false;
    }

    // set file-selectors
    datasetFileHandle.readSelector.columnEnd = 10;
    datasetFileHandle.readSelector.endRow = 10000;
    referenceFileHandle.readSelector.columnStart = 784;
    referenceFileHandle.readSelector.columnEnd = 794;
    referenceFileHandle.readSelector.endRow = 10000;

    // init buffer for output
    std::vector<float> referenceDatasetOutput(10, 0.0f);
    std::vector<float> datasetOutput(10, 0.0f);

    float accuracy = 0.0f;

    // check files
    for (uint64_t row = 0; row < 10000; row++) {
        if (getDataFromDataSet(referenceDatasetOutput, referenceFileHandle, row, error) != OK) {
            status.statusCode = INVALID_INPUT;
            status.errorMessage
                = "Dataset with UUID '" + datasetUuid + "' is invalid and can not be compared";
            error.addMessage(status.errorMessage);
            return false;
        }
        if (getDataFromDataSet(datasetOutput, datasetFileHandle, row, error) != OK) {
            status.statusCode = INVALID_INPUT;
            status.errorMessage = "Dataset with result with UUID '" + referenceUuid
                                  + "' is invalid and can not be checked";
            error.addMessage(status.errorMessage);
            return false;
        }

        bool allCorrect = true;
        setHighest(datasetOutput);
        for (uint64_t i = 0; i < 10; i++) {
            if (referenceDatasetOutput[i] != datasetOutput[i]) {
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
 * @brief set highest output to 1 and other to 0
 *
 * @param outputInterface interface to process
 */
void
CheckMnistDataSetV1M0::setHighest(std::vector<float>& values)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for (uint32_t outputNeuronId = 0; outputNeuronId < values.size(); outputNeuronId++) {
        value = values[outputNeuronId];

        if (value > hightest) {
            hightest = value;
            hightestPos = outputNeuronId;
        }
        values[outputNeuronId] = 0.0f;
    }
    values[hightestPos] = 1.0f;
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
CheckMnistDataSetV1M0::getFileHandle(DataSetFileHandle& fileHandle,
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
