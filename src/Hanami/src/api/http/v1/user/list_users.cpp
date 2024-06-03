/**
 * @file        list_users.cpp
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

#include "list_users.h"

#include <database/users_table.h>
#include <hanami_root.h>

/**
 * @brief constructor
 */
ListUsers::ListUsers() : Blossom("Get information of all registered users.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    json headerMatch = json::array();
    headerMatch.push_back("created_at");
    headerMatch.push_back("id");
    headerMatch.push_back("name");
    headerMatch.push_back("creator_id");
    headerMatch.push_back("projects");
    headerMatch.push_back("is_admin");

    registerOutputField("header", SAKURA_ARRAY_TYPE)
        .setComment("Array with the namings all columns of the table.")
        .setMatch(headerMatch);

    registerOutputField("body", SAKURA_ARRAY_TYPE)
        .setComment("Array with all rows of the table, which array arrays too.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListUsers::runTask(BlossomIO& blossomIO,
                   const json& context,
                   BlossomStatus& status,
                   Hanami::ErrorContainer& error)
{
    // check if admin
    if (context["is_admin"] == false) {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get data from table
    Hanami::TableItem table;
    if (UserTable::getInstance()->getAllUser(table, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    blossomIO.output["header"] = table.getInnerHeader();
    blossomIO.output["body"] = table.getBody();

    return true;
}
