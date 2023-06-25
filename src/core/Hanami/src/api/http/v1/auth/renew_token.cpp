/**
 * @file        renew_token.cpp
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

#include "renew_token.h"

#include <hanami_root.h>

#include <libKitsunemimiCrypto/hashes.h>
#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 */
RenewToken::RenewToken()
    : Blossom("Create a JWT-access-token for a specific user.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("project_id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the project, which has to be used for the new token.");
    assert(addFieldBorder("project_id", 4, 256));
    assert(addFieldRegex("project_id", ID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id",
                        SAKURA_STRING_TYPE,
                        "ID of the user.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the user.");
    registerOutputField("is_admin",
                        SAKURA_BOOL_TYPE,
                        "Set this to true to register the new user as admin.");
    registerOutputField("token",
                        SAKURA_STRING_TYPE,
                        "New JWT-access-token for the user.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
RenewToken::runTask(BlossomIO &blossomIO,
                    const Kitsunemimi::DataMap &context,
                    BlossomStatus &status,
                    Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);
    const std::string projectId = blossomIO.input.get("project_id").getString();

    // get data from table
    Kitsunemimi::JsonItem userData;
    if(UsersTable::getInstance()->getUser(userData, userContext.userId, error, false) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    Kitsunemimi::JsonItem parsedProjects = userData.get("projects");

    // if user is global admin, add the admin-project to the list of choosable projects
    const bool isAdmin = userData.get("is_admin").getBool();
    if(isAdmin)
    {
        Kitsunemimi::DataMap* adminProject = new Kitsunemimi::DataMap();
        adminProject->insert("project_id", new Kitsunemimi::DataValue("admin"));
        adminProject->insert("role", new Kitsunemimi::DataValue("admin"));
        adminProject->insert("is_project_admin", new Kitsunemimi::DataValue(true));
        parsedProjects.append(adminProject);
    }

    // select project
    if(chooseProject(userData, parsedProjects, projectId) == false)
    {
        status.errorMessage = "User with id '"
                              + userContext.userId
                              + "' is not assigned to the project with id '"
                              + projectId
                              + "'.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // create token
    // TODO: make validation-time configurable
    std::string jwtToken;
    if(HanamiRoot::jwt->create_HS256_Token(jwtToken, userData, 3600, error) == false)
    {
        error.addMeesage("Failed to create JWT-Token");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    blossomIO.output.insert("id", userContext.userId);
    blossomIO.output.insert("is_admin", isAdmin);
    blossomIO.output.insert("name", userData.get("name").getString());
    blossomIO.output.insert("token", jwtToken);

    return true;
}

/**
 * @brief get project information for a new selected project from the user-assigned data
 *
 * @param userData user-data coming from database
 * @param parsedProjects list of projects, which are assigned to the user
 * @param selectedProjectId new desired project-id for the new token
 *
 * @return true, if selectedProjectId is available for the user, else false
 */
bool
RenewToken::chooseProject(Kitsunemimi::JsonItem &userData,
                          Kitsunemimi::JsonItem &parsedProjects,
                          const std::string selectedProjectId)
{
    for(uint64_t i = 0; i < parsedProjects.size(); i++)
    {
        if(parsedProjects.get(i).get("project_id").getString() == selectedProjectId)
        {
            userData.insert("project_id", parsedProjects.get(i).get("project_id"));
            userData.insert("role", parsedProjects.get(i).get("role"));
            userData.insert("is_project_admin", parsedProjects.get(i).get("is_project_admin"));
            userData.remove("projects");

            return true;
        }
    }

    return false;
}
