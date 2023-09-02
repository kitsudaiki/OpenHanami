/**
 * @file        list_user_projects.cpp
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

#include "list_user_projects.h"

#include <hanami_root.h>
#include <database/users_table.h>

#include <hanami_common/methods/string_methods.h>
#include <hanami_crypto/hashes.h>
#include <hanami_json/json_item.h>

/**
 * @brief constructor
 */
ListUserProjects::ListUserProjects()
    : Blossom("List all available projects of the user, who made the request.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("user_id", SAKURA_STRING_TYPE)
            .setComment("ID of the user.")
            .setLimit(4, 256)
            .setRegex(ID_EXT_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("projects", SAKURA_ARRAY_TYPE)
            .setComment("Json-array with all assigned projects "
                        "together with role and project-admin-status.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListUserProjects::runTask(BlossomIO &blossomIO,
                          const Kitsunemimi::DataMap &context,
                          BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);
    std::string userId = blossomIO.input.get("user_id").getString();

    // only admin is allowed to request the project-list of other users
    if(userId != ""
            && userContext.isAdmin == false)
    {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // if no other user defined, use the user, who made this request
    if(userId == "") {
        userId = userContext.userId;
    }

    // get data from table
    Kitsunemimi::JsonItem userData;
    if(UsersTable::getInstance()->getUser(userData, userId, error, false) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(userData.size() == 0)
    {
        status.errorMessage = "User with id '" + userId + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // if user is global admin, add the admin-project to the list of choosable projects
    const bool isAdmin = userData.get("is_admin").getBool();
    if(isAdmin)
    {
        Kitsunemimi::DataMap* adminProject = new Kitsunemimi::DataMap();
        adminProject->insert("project_id", new Kitsunemimi::DataValue("admin"));
        adminProject->insert("role", new Kitsunemimi::DataValue("admin"));
        adminProject->insert("is_project_admin", new Kitsunemimi::DataValue(true));
        userData.get("projects").append(adminProject);
    }

    blossomIO.output.insert("projects", userData.get("projects").stealItemContent());

    return true;
}
