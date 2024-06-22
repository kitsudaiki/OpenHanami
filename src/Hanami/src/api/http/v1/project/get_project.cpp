/**
 * @file        get_project.cpp
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

#include "get_project.h"

#include <database/projects_table.h>
#include <hanami_root.h>

/**
 * @brief constructor
 */
GetProject::GetProject() : Blossom("Show information of a specific registered user.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id", SAKURA_STRING_TYPE)
        .setComment("Id of the user.")
        .setLimit(4, 254)
        .setRegex(ID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("created_at", SAKURA_STRING_TYPE)
        .setComment("Timestamp, when project was created.");

    registerOutputField("id", SAKURA_STRING_TYPE).setComment("ID of the new project.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the new project.");

    registerOutputField("creator_id", SAKURA_STRING_TYPE)
        .setComment("Id of the creator of the user.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetProject::runTask(BlossomIO& blossomIO,
                    const json& context,
                    BlossomStatus& status,
                    Hanami::ErrorContainer& error)
{
    // check if admin
    if (context["is_admin"] == false) {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get information from request
    const std::string projectId = blossomIO.input["id"];

    // get data from table
    const ReturnStatus ret
        = ProjectTable::getInstance()->getProject(blossomIO.output, projectId, false, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Project with id '" + projectId + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
