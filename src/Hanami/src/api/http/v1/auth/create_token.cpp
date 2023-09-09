/**
 * @file        create_token.cpp
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

#include "create_token.h"

#include <hanami_root.h>
#include <database/users_table.h>

#include <hanami_crypto/hashes.h>
#include <hanami_config/config_handler.h>

#include <jwt-cpp/jwt.h>
// #include <jwt-cpp/traits/nlohmann-json/defaults.h>

/**
 * @brief constructor
 */
CreateToken::CreateToken()
    : Blossom("Create a JWT-access-token for a specific user.", false)
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id", SAKURA_STRING_TYPE)
            .setComment("ID of the user.")
            .setRegex(ID_EXT_REGEX)
            .setLimit(4, 256);

    registerInputField("password", SAKURA_STRING_TYPE)
            .setComment("Passphrase of the user, to verify the access.")
            .setLimit(8, 4096);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id", SAKURA_STRING_TYPE)
            .setComment("ID of the user.");

    registerOutputField("name", SAKURA_STRING_TYPE)
            .setComment("Name of the user.");

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
CreateToken::runTask(BlossomIO &blossomIO,
                     const json &,
                     BlossomStatus &status,
                     Hanami::ErrorContainer &error)
{
    const std::string userId = blossomIO.input["id"];

    // get data from table
    json userData;
    if(UsersTable::getInstance()->getUser(userData, userId, error, true) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(userData.size() == 0)
    {
        status.errorMessage = "ACCESS DENIED!\n"
                              "User or password is incorrect.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // regenerate password-hash for comparism
    std::string compareHash = "";
    const std::string saltedPw = std::string(blossomIO.input["password"])
                                 + std::string(userData["salt"]);
    Hanami::generate_SHA_256(compareHash, saltedPw);

    // check password
    const std::string pwHash = userData["pw_hash"];
    if(pwHash.size() != compareHash.size()
            || memcmp(pwHash.c_str(), compareHash.c_str(), pwHash.size()) != 0)
    {
        status.errorMessage = "ACCESS DENIED!\n"
                              "User or password is incorrect.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // remove entries, which are NOT allowed to be part of the token
    userData.erase("pw_hash");
    userData.erase("salt");

    json parsedProjects = userData["projects"];

    // get project
    const bool isAdmin = userData["is_admin"];
    if(isAdmin)
    {
        // admin user get alway the admin-project per default
        userData["project_id"] = "admin";
        userData["role"] = "admin";
        userData["is_project_admin"] = true;
        userData.erase("projects");
    }
    else if(parsedProjects.size() != 0)
    {
        // normal user get assigned to first project in their project-list at beginning
        userData["project_id"] = parsedProjects[0]["project_id"];
        userData["role"] = parsedProjects[0]["role"];
        userData["is_project_admin"] = parsedProjects[0]["is_project_admin"];
        userData.erase("projects");
    }
    else
    {
        status.errorMessage = "User with id '" + userId + "' has no project assigned.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get expire-time from config
    bool success = false;
    const u_int32_t expireTime = GET_INT_CONFIG("auth", "token_expire_time", success);
    if(success == false)
    {
        error.addMeesage("Could not read 'token_expire_time' from config of misaki.");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
    }

    std::chrono::system_clock::time_point expireTimePoint = std::chrono::system_clock::now() + std::chrono::seconds(expireTime);

    // create token
    const std::string jwtToken = jwt::create()
                                    .set_type("JWT")
                                    .set_expires_at(expireTimePoint)
                                    .set_payload_claim("user", jwt::claim(userData.dump()))
                                    .sign(jwt::algorithm::hs256{(const char*)HanamiRoot::tokenKey.data()});

    blossomIO.output["id"] = userId;
    blossomIO.output["is_admin"] = isAdmin;
    blossomIO.output["name"] = userData["name"];
    blossomIO.output["token"] = jwtToken;

    return true;
}
