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

#include <database/request_result_table.h>
#include <hanami_root.h>

DeleteRequestResult::DeleteRequestResult() : Blossom("Delete a request-result from shiori.")
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
DeleteRequestResult::runTask(BlossomIO& blossomIO,
                             const json& context,
                             BlossomStatus& status,
                             Hanami::ErrorContainer& error)
{
    const std::string uuid = blossomIO.input["uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    // check if request-result exist within the table
    json result;
    const ReturnStatus ret = RequestResultTable::getInstance()->getRequestResult(
        result, uuid, userContext, false, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Request-result with uuid '" + uuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // delete entry from db
    if (RequestResultTable::getInstance()->deleteRequestResult(uuid, userContext, error) != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
