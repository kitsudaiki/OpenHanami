/**
 * @file        get_user.cpp
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

#include "get_user.h"

#include <hanami_root.h>
#include <database/users_table.h>

#include <hanami_json/json_item.h>

/**
 * @brief constructor
 */
GetUser::GetUser()
    : Blossom("Show information of a specific user.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id", SAKURA_STRING_TYPE)
            .setComment("Id of the user.")
            .setLimit(4, 256)
            .setRegex(ID_EXT_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id", SAKURA_STRING_TYPE)
            .setComment("ID of the user.");

    registerOutputField("name", SAKURA_STRING_TYPE)
            .setComment("Name of the user.");

    registerOutputField("creator_id", SAKURA_STRING_TYPE)
            .setComment("Id of the creator of the user.");

    registerOutputField("is_admin", SAKURA_BOOL_TYPE)
            .setComment("Set this to true to register the new user as admin.");

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
GetUser::runTask(BlossomIO &blossomIO,
                 const Kitsunemimi::DataMap &context,
                 BlossomStatus &status,
                 Kitsunemimi::ErrorContainer &error)
{
    // check if admin
    if(context.getBoolByKey("is_admin") == false)
    {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get information from request
    const std::string userId = blossomIO.input.get("id").getString();

    // get data from table
    if(UsersTable::getInstance()->getUser(blossomIO.output, userId, error, false) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(blossomIO.output.size() == 0)
    {
        status.errorMessage = "User with id '" + userId + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    return true;
}
