/**
 * @file        get_progress_dataset_v1_0.cpp
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

#include "get_progress_dataset_v1_0.h"

#include <core/temp_file_handler.h>
#include <database/dataset_table.h>
#include <database/tempfile_table.h>
#include <hanami_root.h>

GetProgressDataSetV1M0::GetProgressDataSetV1M0()
    : Blossom("Get upload progress of a specific dataset.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset set to delete.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the dataset.");

    registerOutputField("complete", SAKURA_BOOL_TYPE)
        .setComment("True, if all temporary files for complete.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetProgressDataSetV1M0::runTask(BlossomIO& blossomIO,
                                const Hanami::UserContext& userContext,
                                BlossomStatus& status,
                                Hanami::ErrorContainer& error)
{
    const std::string datasetUuid = blossomIO.input["uuid"];

    json databaseOutput;
    ReturnStatus ret = DataSetTable::getInstance()->getDataSet(
        databaseOutput, datasetUuid, userContext, true, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + datasetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get all related tempfiles of the dataset
    std::vector<std::string> relatedUuids;
    if (TempfileTable::getInstance()->getRelatedResourceUuids(
            relatedUuids, "dataset", datasetUuid, userContext, error)
        != OK)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // check if upload complete
    bool isComplete = true;
    for (const std::string& uuid : relatedUuids) {
        Hanami::UploadFileHandle* fileHandle
            = TempFileHandler::getInstance()->getFileHandle(uuid, userContext);
        if (fileHandle->bitBuffer->isComplete() == false) {
            isComplete = false;
        }
    }

    // create output
    blossomIO.output["uuid"] = datasetUuid;
    blossomIO.output["complete"] = isComplete;

    return true;
}
