/**
 * @file        delete_request_result.cpp
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

#include "delete_request_result.h"

#include <hanami_root.h>
#include <database/request_result_table.h>

DeleteRequestResult::DeleteRequestResult()
    : Blossom("Delete a request-result from shiori.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the original request-task, which placed the result in shiori.")
            .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteRequestResult::runTask(BlossomIO &blossomIO,
                             const json &context,
                             BlossomStatus &status,
                             Hanami::ErrorContainer &error)
{
    const std::string uuid = blossomIO.input["uuid"];
    const UserContext userContext(context);

    // check if request-result exist within the table
    json result;
    if(RequestResultTable::getInstance()->getRequestResult(result,
                                                           uuid,
                                                           userContext,
                                                           error,
                                                           false) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(result.size() == 0)
    {
        status.errorMessage = "Request-result with uuid '" + uuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // delete entry from db
    if(RequestResultTable::getInstance()->deleteRequestResult(uuid, userContext, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
