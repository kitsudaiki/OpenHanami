/**
 * @file        delete_data_set.cpp
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

#include "delete_data_set.h"

#include <database/data_set_table.h>
#include <hanami_common/methods/file_methods.h>
#include <hanami_root.h>

DeleteDataSet::DeleteDataSet() : Blossom("Delete a speific data-set.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the data-set to delete.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteDataSet::runTask(BlossomIO& blossomIO,
                       const json& context,
                       BlossomStatus& status,
                       Hanami::ErrorContainer& error)
{
    const std::string dataUuid = blossomIO.input["uuid"];
    const UserContext userContext(context);

    // get location from database
    json result;
    if (DataSetTable::getInstance()->getDataSet(result, dataUuid, userContext, error, true)
        == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (result.size() == 0) {
        status.errorMessage = "Data-set with uuid '" + dataUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get location from response
    const std::string location = result["location"];

    // delete entry from db
    if (DataSetTable::getInstance()->deleteDataSet(dataUuid, userContext, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete local files
    if (Hanami::deleteFileOrDir(location, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
