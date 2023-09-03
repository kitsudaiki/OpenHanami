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

#include "get_audit_log.h"

#include <hanami_root.h>
#include <database/audit_log_table.h>

GetAuditLog::GetAuditLog()
    : Blossom("Get audit-log of a user.")
{
    errorCodes.push_back(UNAUTHORIZED_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------
    registerInputField("user_id", SAKURA_STRING_TYPE)
            .setComment("ID of the user, whos entries are requested. Only an admin is allowed to "
                        "set this values. Any other user get only its own log output based on the "
                        "token-context.")
            .setRequired(false)
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
            .setMatch(new Hanami::DataValue("[\"timestamp\","
                                                 "\"user_id\","
                                                 "\"component\","
                                                 "\"endpoint\","
                                                 "\"request_type\"]"));

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
GetAuditLog::runTask(BlossomIO &blossomIO,
                     const Hanami::DataMap &context,
                     BlossomStatus &status,
                     Hanami::ErrorContainer &error)
{
    const UserContext userContext(context);
    std::string userId = blossomIO.input.get("user_id").getString();
    const uint64_t page = blossomIO.input.get("page").getLong();

    // check that if user-id is set, that the user is also an admin
    if(userContext.isAdmin == false
            && userId.length() != 0)
    {
        status.statusCode = UNAUTHORIZED_RTYPE;
        status.errorMessage = "'user_id' can only be set by an admin";
        return false;
    }

    // if no user-id was defined, use the id of the context
    if(userId.length() == 0) {
        userId = userContext.userId;
    }

    // get data from table
    Hanami::TableItem table;
    if(AuditLogTable::getInstance()->getAllAuditLogEntries(table, userId, page, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // create output
    Hanami::DataArray* headerInfo = table.getInnerHeader();
    blossomIO.output.insert("header", headerInfo);
    blossomIO.output.insert("body", table.getBody());
    delete headerInfo;

    return true;
}
