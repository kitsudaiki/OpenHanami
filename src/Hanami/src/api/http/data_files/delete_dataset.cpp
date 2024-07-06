/**
 * @file        delete_dataset.cpp
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

#include "delete_dataset.h"

#include <database/dataset_table.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_root.h>

DeleteDataSetV1M0::DeleteDataSetV1M0() : Blossom("Delete a speific dataset.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the dataset to delete.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteDataSetV1M0::runTask(BlossomIO& blossomIO,
                           const json& context,
                           BlossomStatus& status,
                           Hanami::ErrorContainer& error)
{
    const std::string dataUuid = blossomIO.input["uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    // get location from database
    DataSetTable::DataSetDbEntry result;
    const ReturnStatus ret
        = DataSetTable::getInstance()->getDataSet(result, dataUuid, userContext, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + dataUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete entry from db
    if (DataSetTable::getInstance()->deleteDataSet(dataUuid, userContext, error) != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete local files
    if (Hanami::deleteFileOrDir(result.location, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
