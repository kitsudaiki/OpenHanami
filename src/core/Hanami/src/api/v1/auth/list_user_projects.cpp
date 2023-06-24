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

#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCrypto/hashes.h>
#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/defines.h>
#include <libKitsunemimiHanamiCommon/structs.h>

using namespace Kitsunemimi::Hanami;

/**
 * @brief constructor
 */
ListUserProjects::ListUserProjects()
    : Blossom("List all available projects of the user, who made the request.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("user_id",
                       SAKURA_STRING_TYPE,
                       false,
                       "ID of the user.");
    assert(addFieldBorder("user_id", 4, 256));
    assert(addFieldRegex("user_id", ID_EXT_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("projects",
                        SAKURA_ARRAY_TYPE,
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
ListUserProjects::runTask(BlossomIO &blossomIO,
                          const Kitsunemimi::DataMap &context,
                          BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error)
{
    const Kitsunemimi::Hanami::UserContext userContext(context);
    std::string userId = blossomIO.input.get("user_id").getString();

    // only admin is allowed to request the project-list of other users
    if(userId != ""
            && userContext.isAdmin == false)
    {
        status.statusCode = Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE;
        return false;
    }

    // if no other user defined, use the user, who made this request
    if(userId == "") {
        userId = userContext.userId;
    }

    // get data from table
    Kitsunemimi::JsonItem userData;
    if(HanamiRoot::usersTable->getUser(userData, userId, error, false) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
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
