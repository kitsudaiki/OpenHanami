/**
 * @file        get_progress_data_set.cpp
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

#include "get_progress_data_set.h"

#include <database/data_set_table.h>
#include <hanami_files/data_set_files/data_set_file.h>
#include <hanami_files/data_set_files/image_data_set_file.h>
#include <hanami_files/data_set_files/table_data_set_file.h>
#include <hanami_root.h>

GetProgressDataSet::GetProgressDataSet() : Blossom("Get upload progress of a specific data-set.")
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

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the data-set.");

    registerOutputField("temp_files", SAKURA_MAP_TYPE)
        .setComment("Map with the uuids of the temporary files and it's upload progress");

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
    const std::string dataUuid = blossomIO.input["uuid"];
    const UserContext userContext(context);

    json databaseOutput;
    if (DataSetTable::getInstance()->getDataSet(databaseOutput, dataUuid, userContext, error, true)
        == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (databaseOutput.size() == 0) {
        status.errorMessage = "Data-set with uuid '" + dataUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // parse and add temp-file-information
    const json tempFiles = databaseOutput["temp_files"];

    // check and add if complete
    bool finishedAll = true;
    for (const auto& [key, file] : tempFiles.items()) {
        if (file < 1.0f) {
            finishedAll = false;
        }
    }

    // create output
    blossomIO.output["uuid"] = databaseOutput["uuid"];
    blossomIO.output["temp_files"] = tempFiles;
    blossomIO.output["complete"] = finishedAll;

    return true;
}
