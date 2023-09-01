/**
 * @file        get_request_result.cpp
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

#include "get_request_result.h"

#include <hanami_root.h>
#include <database/request_result_table.h>

GetRequestResult::GetRequestResult()
    : Blossom("Get a specific request-result")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the original request-task, which placed the result in shiori.")
            .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the request-result.");

    registerOutputField("name", SAKURA_STRING_TYPE)
            .setComment("Name of the request-result.");

    registerOutputField("owner_id", SAKURA_STRING_TYPE)
            .setComment("ID of the user, who created the request-result.");

    registerOutputField("project_id", SAKURA_STRING_TYPE)
            .setComment("ID of the project, where the request-result belongs to.");

    registerOutputField("visibility", SAKURA_STRING_TYPE)
            .setComment("Visibility of the request-result (private, shared, public).");

    registerOutputField("data", SAKURA_ARRAY_TYPE)
            .setComment("Result of the request-task as json-array.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetRequestResult::runTask(BlossomIO &blossomIO,
                          const Kitsunemimi::DataMap &context,
                          BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error)
{
    const std::string uuid = blossomIO.input.get("uuid").getString();
    const UserContext userContext(context);

    // check if request-result exist within the table
    if(RequestResultTable::getInstance()->getRequestResult(blossomIO.output,
                                                           uuid,
                                                           userContext,
                                                           error,
                                                           true) == false)
    {
        status.errorMessage = "Request-result with UUID '" + uuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    return true;
}
