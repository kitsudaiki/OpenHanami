/**
 * @file        delete_checkpoint.cpp
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

#include "delete_checkpoint.h"

#include <hanami_root.h>
#include <database/checkpoint_table.h>

#include <hanami_json/json_item.h>
#include <hanami_common/methods/file_methods.h>

DeleteCheckpoint::DeleteCheckpoint()
    : Blossom("Delete a result-set from shiori.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the checkpoint to delete.")
            .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteCheckpoint::runTask(BlossomIO &blossomIO,
                          const Kitsunemimi::DataMap &context,
                          BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error)
{
    const std::string checkpointUuid = blossomIO.input.get("uuid").getString();
    const UserContext userContext(context);

    // get location from database
    Kitsunemimi::JsonItem result;
    if(CheckpointTable::getInstance()->getCheckpoint(result,
                                                     checkpointUuid,
                                                     userContext,
                                                     error,
                                                     true) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(result.size() == 0)
    {
        status.errorMessage = "Chekckpoint with uuid '" + checkpointUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get location from response
    const std::string location = result.get("location").getString();

    // delete entry from db
    if(CheckpointTable::getInstance()->deleteCheckpoint(checkpointUuid,
                                                        userContext,
                                                        error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete local files
    if(Kitsunemimi::deleteFileOrDir(location, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
