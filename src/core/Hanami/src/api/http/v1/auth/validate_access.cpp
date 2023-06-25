/**
 * @file        validate_access.cpp
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

#include "validate_access.h"

#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiHanamiPolicies/policy.h>
#include <hanami_root.h>

using Kitsunemimi::Hanami::HttpRequestType;

/**
 * @brief constructor
 */
ValidateAccess::ValidateAccess()
    : Blossom("Checks if a JWT-access-token of a user is valid or not "
              "and optional check if the user is allowed by its roles "
              "and the policy to access a specific endpoint.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("token",
                       SAKURA_STRING_TYPE,
                       true,
                       "User specific JWT-access-token.");
    assert(addFieldRegex("token", "[a-zA-Z_.\\-0-9]*"));

    registerInputField("component",
                       SAKURA_STRING_TYPE,
                       false,
                       "Requested component-name of the request. If this is not set, then only "
                       "the token in itself will be validated.");
    assert(addFieldBorder("component", 4, 256));
    assert(addFieldRegex("component", "[a-zA-Z][a-zA-Z_0-9]*"));

    registerInputField("endpoint",
                       SAKURA_STRING_TYPE,
                       false,
                       "Requesed endpoint within the component.");
    assert(addFieldBorder("endpoint", 4, 256));
    assert(addFieldRegex("endpoint", "[a-zA-Z][a-zA-Z_/0-9]*"));

    registerInputField("http_type",
                       SAKURA_INT_TYPE,
                       false,
                       "Type of the HTTP-request as enum "
                       "(DELETE = 1, GET = 2, HEAD = 3, POST = 4, PUT = 5).");
    assert(addFieldBorder("http_type", 1, 5));

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
                        "Show if the user is an admin or not.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "Selected project of the user.");
    registerOutputField("role",
                        SAKURA_STRING_TYPE,
                        "Role of the user within the project.");
    registerOutputField("is_project_admin",
                        SAKURA_BOOL_TYPE,
                        "True, if the user is admin within the selected project.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ValidateAccess::runTask(BlossomIO &blossomIO,
                        const Kitsunemimi::DataMap &,
                        BlossomStatus &status,
                        Kitsunemimi::ErrorContainer &error)
{
    // collect information from the input
    const std::string token = blossomIO.input.get("token").getString();
    const std::string component = blossomIO.input.get("component").getString();
    const std::string endpoint = blossomIO.input.get("endpoint").getString();

    // validate token
    std::string publicError;
    if(HanamiRoot::jwt->validateToken(blossomIO.output, token, publicError, error) == false)
    {
        error.addMeesage("Misaki failed to validate JWT-Token");
        status.errorMessage = publicError;
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // allow skipping policy-check
    // TODO: find better solution to make a difference, if policy should be checked or not
    if(component != "")
    {
        if(blossomIO.input.contains("http_type") == false)
        {
            error.addMeesage("http_type is missing in token-request");
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            return false;
        }

        const uint32_t httpTypeValue = blossomIO.input.get("http_type").getInt();
        const HttpRequestType httpType = static_cast<HttpRequestType>(httpTypeValue);

        // check policy
        const std::string role = blossomIO.output.get("role").getString();
        if(Policy::getInstance()->checkUserAgainstPolicy(endpoint, httpType, role) == false)
        {
            status.errorMessage = "Access denied by policy";
            status.statusCode = UNAUTHORIZED_RTYPE;
            error.addMeesage(status.errorMessage);
            return false;
        }
    }

    // remove irrelevant fields
    blossomIO.output.remove("pw_hash");
    blossomIO.output.remove("creator_id");
    blossomIO.output.remove("exp");
    blossomIO.output.remove("iat");
    blossomIO.output.remove("nbf");

    return true;
}

