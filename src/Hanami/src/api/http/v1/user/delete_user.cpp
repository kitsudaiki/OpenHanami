/**
 * @file        delete_user.cpp
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

#include "delete_user.h"

#include <hanami_root.h>

#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 */
DeleteUser::DeleteUser()
    : Blossom("Delete a specific user from the database.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the user.");
    // column in database is limited to 256 characters size
    assert(addFieldBorder("id", 4, 256));
    assert(addFieldRegex("id", ID_EXT_REGEX));

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteUser::runTask(BlossomIO &blossomIO,
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
    const std::string deleterId = context.getStringByKey("id");
    const std::string userId = blossomIO.input.get("id").getString();

    // check if user exist within the table
    Kitsunemimi::JsonItem result;
    if(UsersTable::getInstance()->getUser(result, userId, error, false) == false)
    {
        status.errorMessage = "User with id '" + userId + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // prevent user from deleting himself
    if(result.get("id").getString() == deleterId)
    {
        status.errorMessage = "User with id '"
                              + userId
                              + "' tries to delete himself, which is not allowed.";
        status.statusCode = BAD_REQUEST_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get data from table
    if(UsersTable::getInstance()->deleteUser(userId, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
