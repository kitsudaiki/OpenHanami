/**
 * @file        get_checkpoint.cpp
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

#include "get_checkpoint.h"

#include <hanami_root.h>
#include <database/checkpoint_table.h>

GetCheckpoint::GetCheckpoint()
    : Blossom("Get checkpoint of a cluster.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the original request-task, which placed the result in shiori.")
            .setRegex("[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-"
                      "[a-fA-F0-9]{4}-[a-fA-F0-9]{12}");

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the data-set.");

    registerOutputField("name", SAKURA_STRING_TYPE)
            .setComment("Name of the data-set.");

    registerOutputField("location", SAKURA_STRING_TYPE)
            .setComment("File path on local storage.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetCheckpoint::runTask(BlossomIO &blossomIO,
                       const Kitsunemimi::DataMap &context,
                       BlossomStatus &status,
                       Kitsunemimi::ErrorContainer &error)
{
    const std::string checkpointUuid = blossomIO.input.get("uuid").getString();
    const UserContext userContext(context);

    // get checkpoint-info from database
    if(CheckpointTable::getInstance()->getCheckpoint(blossomIO.output,
                                                     checkpointUuid,
                                                     userContext,
                                                     error,
                                                     true) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(blossomIO.output.size() == 0)
    {
        status.errorMessage = "Checkpoint with uuid '" + checkpointUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // remove irrelevant fields
    blossomIO.output.remove("owner_id");
    blossomIO.output.remove("project_id");
    blossomIO.output.remove("visibility");
    blossomIO.output.remove("temp_files");

    return true;
}
