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

#include <libKitsunemimiCrypto/hashes.h>
#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiConfig/config_handler.h>

/**
 * @brief constructor
 */
CreateToken::CreateToken()
    : Blossom("Create a JWT-access-token for a specific user.", false)
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the user.");
    assert(addFieldBorder("id", 4, 256));
    assert(addFieldRegex("id", ID_EXT_REGEX));

    registerInputField("password",
                       SAKURA_STRING_TYPE,
                       true,
                       "Passphrase of the user, to verify the access.");
    assert(addFieldBorder("password", 8, 4096));

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
CreateToken::runTask(BlossomIO &blossomIO,
                     const Kitsunemimi::DataMap &,
                     BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error)
{
    const std::string userId = blossomIO.input.get("id").getString();

    // get data from table
    Kitsunemimi::JsonItem userData;
    if(UsersTable::getInstance()->getUser(userData, userId, error, true) == false)
    {
        status.errorMessage = "ACCESS DENIED!\n"
                              "User or password is incorrect.";
        error.addMeesage(status.errorMessage);
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // regenerate password-hash for comparism
    std::string compareHash = "";
    const std::string saltedPw = blossomIO.input.get("password").getString()
                                 + userData.get("salt").getString();
    Kitsunemimi::generate_SHA_256(compareHash, saltedPw);

    // check password
    const std::string pwHash = userData.get("pw_hash").getString();
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
    std::string jwtToken;
    userData.remove("pw_hash");
    userData.remove("salt");

    Kitsunemimi::JsonItem parsedProjects = userData.get("projects");

    // get project
    const bool isAdmin = userData.get("is_admin").getBool();
    if(isAdmin)
    {
        // admin user get alway the admin-project per default
        userData.insert("project_id", "admin");
        userData.insert("role", "admin");
        userData.insert("is_project_admin", true);
        userData.remove("projects");
    }
    else if(parsedProjects.size() != 0)
    {
        // normal user get assigned to first project in their project-list at beginning
        userData.insert("project_id", parsedProjects.get(0).get("project_id"));
        userData.insert("role", parsedProjects.get(0).get("role"));
        userData.insert("is_project_admin", parsedProjects.get(0).get("is_project_admin"));
        userData.remove("projects");
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

    // create token
    if(HanamiRoot::jwt->create_HS256_Token(jwtToken, userData, expireTime, error) == false)
    {
        error.addMeesage("Failed to create JWT-Token");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    blossomIO.output.insert("id", userId);
    blossomIO.output.insert("is_admin", isAdmin);
    blossomIO.output.insert("name", userData.get("name").getString());
    blossomIO.output.insert("token", jwtToken);

    return true;
}
