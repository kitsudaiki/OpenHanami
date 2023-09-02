/**
 * @file        get_audit_log.cpp
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

#include "get_error_log.h"

#include <hanami_root.h>
#include <database/error_log_table.h>

GetErrorLog::GetErrorLog()
    : Blossom("Get error-log of a user. Only an admin is allowed to request the error-log.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------
    registerInputField("user_id", SAKURA_STRING_TYPE)
            .setComment("ID of the user, whos entries are requested.")
            .setDefault(new Kitsunemimi::DataValue(""))
            .setLimit(4, 256)
            .setRegex(ID_EXT_REGEX);

    registerInputField("page", SAKURA_INT_TYPE)
            .setComment("Page-number starting with 0 to access the logs. "
                       "A page has up to 100 entries.")
            .setLimit(0, 1000000000);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("header", SAKURA_ARRAY_TYPE)
            .setComment("Array with the namings all columns of the table.")
            .setMatch(new Kitsunemimi::DataValue("[\"timestamp\","
                                                 "\"user_id\","
                                                 "\"component\","
                                                 "\"context\","
                                                 "\"input_values\","
                                                 "\"message\"]"));

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
GetErrorLog::runTask(BlossomIO &blossomIO,
                     const Kitsunemimi::DataMap &context,
                     BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);
    const uint64_t page = blossomIO.input.get("page").getLong();

    // check that the user is an admin
    if(userContext.isAdmin == false)
    {
        status.statusCode = UNAUTHORIZED_RTYPE;
        status.errorMessage = "only an admin is allowed to request error-logs";
        return false;
    }

    const std::string userId = blossomIO.input.get("user_id").getString();

    // get data from table
    Kitsunemimi::TableItem table;
    if(ErrorLogTable::getInstance()->getAllErrorLogEntries(table, userId, page, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // create output
    Kitsunemimi::DataArray* headerInfo = table.getInnerHeader();
    blossomIO.output.insert("header", headerInfo);
    blossomIO.output.insert("body", table.getBody());
    delete headerInfo;

    return true;
}
