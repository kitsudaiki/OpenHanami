/**
 * @file        list_user_projects_v1_0.cpp
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

#include "list_user_projects_v1_0.h"

#include <database/users_table.h>
#include <hanami_common/functions/string_functions.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>

/**
 * @brief constructor
 */
ListUserProjectsV1M0::ListUserProjectsV1M0()
    : Blossom("List all available projects of the user, who made the request.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("user_id", SAKURA_STRING_TYPE)
        .setComment("ID of the user.")
        .setLimit(4, 254)
        .setRegex(ID_EXT_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("projects", SAKURA_ARRAY_TYPE)
        .setComment(
            "Json-array with all assigned projects "
            "together with role and project-admin-status.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListUserProjectsV1M0::runTask(BlossomIO& blossomIO,
                              const Hanami::UserContext& userContext,
                              BlossomStatus& status,
                              Hanami::ErrorContainer& error)
{
    std::string userId = blossomIO.input["user_id"];

    // only admin is allowed to request the project-list of other users
    if (userId != "" && userContext.isAdmin == false) {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // if no other user defined, use the user, who made this request
    if (userId == "") {
        userId = userContext.userId;
    }

    // get data from table
    json userData;
    const ReturnStatus ret = UserTable::getInstance()->getUser(userData, userId, false, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "User with id '" + userId + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // if user is global admin, add the admin-project to the list of choosable projects
    const bool isAdmin = userData["is_admin"];
    if (isAdmin) {
        json adminProject = json::object();
        adminProject["project_id"] = "admin";
        adminProject["role"] = "admin";
        adminProject["is_project_admin"] = true;
        userData["projects"] = adminProject;
    }

    blossomIO.output["projects"] = userData["projects"];

    return true;
}
