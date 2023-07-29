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

#include <hanami_root.h>

/**
 * @brief constructor
 */
ListUsers::ListUsers()
    : Blossom("Get information of all registered users.")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("header",
                        SAKURA_ARRAY_TYPE,
                        "Array with the namings all columns of the table.");
    assert(addFieldMatch("header", new Kitsunemimi::DataValue("[\"id\","
                                                              "\"name\","
                                                              "\"creator_id\","
                                                              "\"projects\","
                                                              "\"is_admin\"]")));
    registerOutputField("body",
                        SAKURA_ARRAY_TYPE,
                        "Array with all rows of the table, which array arrays too.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListUsers::runTask(BlossomIO &blossomIO,
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

    // get data from table
    Kitsunemimi::TableItem table;
    if(UsersTable::getInstance()->getAllUser(table, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    Kitsunemimi::DataArray* headerInfo = table.getInnerHeader();
    blossomIO.output.insert("header", headerInfo);
    blossomIO.output.insert("body", table.getBody());
    delete headerInfo;

    return true;
}
