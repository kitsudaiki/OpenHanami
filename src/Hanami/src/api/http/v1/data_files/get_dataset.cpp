/**
 * @file        get_dataset.cpp
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

#include "get_dataset.h"

#include <database/dataset_table.h>
#include <hanami_files/dataset_files/dataset_functions.h>
#include <hanami_root.h>

GetDataSet::GetDataSet() : Blossom("Get information of a specific dataset.")
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

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the dataset.");

    registerOutputField("owner_id", SAKURA_STRING_TYPE)
        .setComment("ID of the user, who created the dataset.");

    registerOutputField("project_id", SAKURA_STRING_TYPE)
        .setComment("ID of the project, where the dataset belongs to.");

    registerOutputField("visibility", SAKURA_STRING_TYPE)
        .setComment("Visibility of the dataset (private, shared, public).");

    registerOutputField("location", SAKURA_STRING_TYPE)
        .setComment("Local file-path of the dataset.");

    registerOutputField("type", SAKURA_STRING_TYPE)
        .setComment("Type of the new set (csv or mnist)");

    registerOutputField("inputs", SAKURA_INT_TYPE).setComment("Number of inputs.");

    registerOutputField("outputs", SAKURA_INT_TYPE).setComment("Number of outputs.");

    registerOutputField("lines", SAKURA_INT_TYPE).setComment("Number of lines.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetDataSet::runTask(BlossomIO& blossomIO,
                    const json& context,
                    BlossomStatus& status,
                    Hanami::ErrorContainer& error)
{
    const std::string dataUuid = blossomIO.input["uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    const ReturnStatus ret = DataSetTable::getInstance()->getDateSetInfo(
        blossomIO.output, dataUuid, userContext, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + dataUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    return true;
}
