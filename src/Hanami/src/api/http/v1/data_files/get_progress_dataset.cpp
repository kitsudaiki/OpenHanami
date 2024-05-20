/**
 * @file        get_progress_dataset.cpp
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

#include "get_progress_dataset.h"

#include <core/temp_file_handler.h>
#include <database/dataset_table.h>
#include <database/tempfile_table.h>
#include <hanami_files/dataset_files/dataset_file.h>
#include <hanami_files/dataset_files/image_dataset_file.h>
#include <hanami_files/dataset_files/table_dataset_file.h>
#include <hanami_root.h>

GetProgressDataSet::GetProgressDataSet() : Blossom("Get upload progress of a specific dataset.")
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
GetProgressDataSet::runTask(BlossomIO& blossomIO,
                            const json& context,
                            BlossomStatus& status,
                            Hanami::ErrorContainer& error)
{
    const std::string datasetUuid = blossomIO.input["uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    json databaseOutput;
    if (DataSetTable::getInstance()->getDataSet(
            databaseOutput, datasetUuid, userContext, error, true)
        == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (databaseOutput.size() == 0) {
        status.errorMessage = "Data-set with uuid '" + datasetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get all related tempfiles of the dataset
    std::vector<std::string> relatedUuids;
    if (TempfileTable::getInstance()->getRelatedResourceUuids(
            relatedUuids, "dataset", datasetUuid, userContext, error)
        == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // check if upload complete
    bool isComplete = true;
    for (const std::string& uuid : relatedUuids) {
        Hanami::FileHandle* fileHandle
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
