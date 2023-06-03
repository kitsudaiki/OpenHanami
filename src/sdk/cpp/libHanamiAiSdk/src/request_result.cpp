/**
 * @file        request_result.cpp
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

#include <libHanamiAiSdk/request_result.h>
#include <common/http_client.h>

namespace HanamiAI
{

/**
 * @brief get results of a request from shiori
 *
 * @param result reference for response-message
 * @param requestResultUuid uuid of the requested result
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getRequestResult(std::string &result,
                 const std::string &requestResultUuid,
                 Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/request_result";
    const std::string vars = "uuid=" + requestResultUuid;

    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get request-result with uuid '" + requestResultUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible request-results
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listRequestResult(std::string &result,
                  Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/request_result/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list request-results");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a user from misaki
 *
 * @param result reference for response-message
 * @param requestResultUuid uuid of the request-result, which should be deleted
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteRequestResult(std::string &result,
                    const std::string &requestResultUuid,
                    Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/request_result";
    const std::string vars = "uuid=" + requestResultUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete request-result with uuid '" + requestResultUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace HanamiAI
