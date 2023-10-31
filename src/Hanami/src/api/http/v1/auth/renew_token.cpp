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

#include <database/users_table.h>
#include <hanami_config/config_handler.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>
#include <jwt-cpp/jwt.h>
// #include <jwt-cpp/traits/nlohmann-json/defaults.h>

/**
 * @brief constructor
 */
RenewToken::RenewToken() : Blossom("Renew a JWT-access-token for a specific user.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("project_id", SAKURA_STRING_TYPE)
        .setComment("ID of the project, which has to be used for the new token.")
        .setLimit(4, 256)
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
 * @brief runTask
 */
bool
RenewToken::runTask(BlossomIO& blossomIO,
                    const json& context,
                    BlossomStatus& status,
                    Hanami::ErrorContainer& error)
{
    const UserContext userContext(context);
    const std::string projectId = blossomIO.input["project_id"];

    // get data from table
    json userData;
    if (UsersTable::getInstance()->getUser(userData, userContext.userId, error, false) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (userData.size() == 0) {
        status.errorMessage
            = "ACCESS DENIED!\n"
              "User or password is incorrect.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    json parsedProjects = userData["projects"];

    // if user is global admin, add the admin-project to the list of choosable projects
    const bool isAdmin = userData["is_admin"];
    if (isAdmin) {
        json adminProject = json::object();
        adminProject["project_id"] = "admin";
        adminProject["role"] = "admin";
        adminProject["is_project_admin"] = true;
        parsedProjects.push_back(adminProject);
    }

    // select project
    if (chooseProject(userData, parsedProjects, projectId) == false) {
        status.errorMessage = "User with id '" + userContext.userId
                              + "' is not assigned to the project with id '" + projectId + "'.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get expire-time from config
    bool success = false;
    const u_int32_t expireTime = GET_INT_CONFIG("auth", "token_expire_time", success);
    if (success == false) {
        error.addMeesage("Could not read 'token_expire_time' from config of misaki.");
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
              .set_payload_claim("user", jwt::claim(userData.dump()))
              .sign(jwt::algorithm::hs256{(const char*)HanamiRoot::tokenKey.data()});

    blossomIO.output["id"] = userContext.userId;
    blossomIO.output["is_admin"] = isAdmin;
    blossomIO.output["name"] = userData["name"];
    blossomIO.output["token"] = jwtToken;

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
RenewToken::chooseProject(json& userData, json& parsedProjects, const std::string selectedProjectId)
{
    for (uint64_t i = 0; i < parsedProjects.size(); i++) {
        if (parsedProjects[i]["project_id"] == selectedProjectId) {
            userData["project_id"] = parsedProjects[i]["project_id"];
            userData["role"] = parsedProjects[i]["role"];
            userData["is_project_admin"] = parsedProjects[i]["is_project_admin"];
            userData.erase("projects");

            return true;
        }
    }

    return false;
}
