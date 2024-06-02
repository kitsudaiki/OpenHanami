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

#include <database/users_table.h>
#include <hanami_root.h>

/**
 * @brief constructor
 */
DeleteUser::DeleteUser() : Blossom("Delete a specific user from the database.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id", SAKURA_STRING_TYPE)
        .setComment("ID of the user.")
        .setLimit(4, 256)
        .setRegex(ID_EXT_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteUser::runTask(BlossomIO& blossomIO,
                    const json& context,
                    BlossomStatus& status,
                    Hanami::ErrorContainer& error)
{
    // check if admin
    if (context["is_admin"] == false) {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get information from request
    const std::string deleterId = context["id"];
    const std::string userId = blossomIO.input["id"];

    // prevent user from deleting himself
    if (userId == deleterId) {
        status.errorMessage
            = "User with id '" + userId + "' tries to delete himself, which is not allowed.";
        status.statusCode = CONFLICT_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // delete data from table
    const ReturnStatus ret = UserTable::getInstance()->deleteUser(userId, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "User with id '" + userId + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
