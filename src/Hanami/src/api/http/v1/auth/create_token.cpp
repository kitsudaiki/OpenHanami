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

#include <database/users_table.h>
#include <hanami_config/config_handler.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>
#include <jwt-cpp/jwt.h>
// #include <jwt-cpp/traits/nlohmann-json/defaults.h>

/**
 * @brief constructor
 */
CreateToken::CreateToken() : Blossom("Create a JWT-access-token for a specific user.", false)
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
CreateToken::runTask(BlossomIO& blossomIO,
                     const json&,
                     BlossomStatus& status,
                     Hanami::ErrorContainer& error)
{
    const std::string userId = blossomIO.input["id"];
    const std::string password = blossomIO.input["password"];

    // get data from table
    UserTable::UserDbEntry userData;
    ReturnStatus ret = UserTable::getInstance()->getUser(userData, userId, error);
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

    // regenerate password-hash for comparism
    std::string compareHash = "";
    const std::string saltedPw = password + userData.salt;
    Hanami::generate_SHA_256(compareHash, saltedPw);

    // check password
    const std::string pwHash = userData.pwHash;
    if (pwHash.size() != compareHash.size()
        || memcmp(pwHash.c_str(), compareHash.c_str(), pwHash.size()) != 0)
    {
        status.errorMessage
            = "ACCESS DENIED!\n"
              "User or password is incorrect.";
        status.statusCode = UNAUTHORIZED_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get reduced user-data as json from db
    json tokenUserData;
    ret = UserTable::getInstance()->getUser(tokenUserData, userId, false, error);
    if (ret != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get project
    if (userData.isAdmin) {
        // admin user get alway the admin-project per default
        tokenUserData["project_id"] = "admin";
        tokenUserData["role"] = "admin";
        tokenUserData["is_project_admin"] = true;
        tokenUserData.erase("projects");
    }
    else if (userData.projects.size() > 0) {
        // normal user get assigned to first project in their project-list at beginning
        tokenUserData["project_id"] = userData.projects.at(0).projectId;
        tokenUserData["role"] = userData.projects.at(0).role;
        tokenUserData["is_project_admin"] = userData.projects.at(0).isProjectAdmin;
        tokenUserData.erase("projects");
    }
    else {
        status.errorMessage = "User with id '" + userId + "' has no project assigned.";
        status.statusCode = UNAUTHORIZED_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get expire-time from config
    bool success = false;
    const u_int32_t expireTime = GET_INT_CONFIG("auth", "token_expire_time", success);
    if (success == false) {
        error.addMessage("Could not read 'token_expire_time' from config of hanami.");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
    }

    std::chrono::system_clock::time_point expireTimePoint
        = std::chrono::system_clock::now() + std::chrono::seconds(expireTime);

    // create token
    const std::string jwtToken
        = jwt::create()
              .set_type("JWT")
              .set_expires_at(expireTimePoint)
              .set_payload_claim("user", jwt::claim(tokenUserData.dump()))
              .sign(jwt::algorithm::hs256{(const char*)HanamiRoot::tokenKey.data()});

    blossomIO.output["id"] = userId;
    blossomIO.output["is_admin"] = userData.isAdmin;
    blossomIO.output["name"] = userData.name;
    blossomIO.output["token"] = jwtToken;

    return true;
}
