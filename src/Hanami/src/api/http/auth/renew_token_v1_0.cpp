/**
 * @file        renew_token_v1_0.cpp
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

#include "renew_token_v1_0.h"

#include <database/users_table.h>
#include <hanami_config/config_handler.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>
#include <jwt-cpp/jwt.h>
// #include <jwt-cpp/traits/nlohmann-json/defaults.h>

/**
 * @brief constructor
 */
RenewTokenV1M0::RenewTokenV1M0() : Blossom("Renew a JWT-access-token for a specific user.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("project_id", SAKURA_STRING_TYPE)
        .setComment("ID of the project, which has to be used for the new token.")
        .setLimit(4, 254)
        .setRegex(ID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id", SAKURA_STRING_TYPE).setComment("ID of the user.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the user.");

    registerOutputField("is_admin", SAKURA_BOOL_TYPE)
        .setComment("Set this to true to register the new user as admin.");

    registerOutputField("token", SAKURA_STRING_TYPE)
        .setComment("New JWT-access-token for the user.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief get project information for a new selected project from the user-assigned data
 *
 * @param tokenUserData user-data coming from database
 * @param parsedProjects list of projects, which are assigned to the user
 * @param selectedProjectId new desired project-id for the new token
 *
 * @return true, if selectedProjectId is available for the user, else false
 */
bool
chooseProject(json& tokenUserData,
              const std::vector<UserTable::UserProjectDbEntry>& parsedProjects,
              const std::string selectedProjectId)
{
    for (const UserTable::UserProjectDbEntry project : parsedProjects) {
        if (project.projectId == selectedProjectId) {
            tokenUserData["project_id"] = project.projectId;
            tokenUserData["role"] = project.role;
            tokenUserData["is_project_admin"] = project.isProjectAdmin;
            tokenUserData.erase("projects");

            return true;
        }
    }

    return false;
}

/**
 * @brief runTask
 */
bool
RenewTokenV1M0::runTask(BlossomIO& blossomIO,
                        const json& context,
                        BlossomStatus& status,
                        Hanami::ErrorContainer& error)
{
    const Hanami::UserContext userContext = convertContext(context);
    const std::string projectId = blossomIO.input["project_id"];

    // get data from table
    UserTable::UserDbEntry userData;
    ReturnStatus ret = UserTable::getInstance()->getUser(userData, userContext.userId, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage
            = "ACCESS DENIED!\n"
              "User or password is incorrect.";
        status.statusCode = UNAUTHORIZED_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get reduced user-data as json from db
    json tokenUserData;
    ret = UserTable::getInstance()->getUser(tokenUserData, userContext.userId, false, error);
    if (ret != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // if user is global admin, add the admin-project to the list of choosable projects
    if (userData.isAdmin) {
        UserTable::UserProjectDbEntry adminProject;
        adminProject.projectId = "admin";
        adminProject.role = "admin";
        adminProject.isProjectAdmin = true;
        userData.projects.push_back(adminProject);
    }

    // select project
    if (chooseProject(tokenUserData, userData.projects, projectId) == false) {
        status.errorMessage = "User with id '" + userContext.userId
                              + "' is not assigned to the project with id '" + projectId + "'.";
        status.statusCode = UNAUTHORIZED_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get expire-time from config
    bool success = false;
    const u_int32_t expireTime = GET_INT_CONFIG("auth", "token_expire_time", success);
    if (success == false) {
        error.addMessage("Could not read 'token_expire_time' from config of misaki.");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
    }

    std::chrono::system_clock::time_point expireTimePoint
        = std::chrono::system_clock::now() + std::chrono::seconds(expireTime);

    // create token
    // TODO: make validation-time configurable

    const std::string jwtToken
        = jwt::create()
              .set_type("JWT")
              .set_expires_at(expireTimePoint)
              .set_payload_claim("user", jwt::claim(tokenUserData.dump()))
              .sign(jwt::algorithm::hs256{(const char*)HanamiRoot::tokenKey.data()});

    blossomIO.output["id"] = userContext.userId;
    blossomIO.output["is_admin"] = userData.isAdmin;
    blossomIO.output["name"] = userData.name;
    blossomIO.output["token"] = jwtToken;
    blossomIO.output.erase("created_at");

    return true;
}
