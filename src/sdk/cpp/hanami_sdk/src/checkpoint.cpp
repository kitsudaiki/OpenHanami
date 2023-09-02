/**
 * @file        checkpoint.cpp
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

#include <hanami_sdk/checkpoint.h>
#include <common/http_client.h>

namespace Hanami
{

/**
 * @brief get information of a checkpoint from shiori
 *
 * @param result reference for response-message
 * @param checkpointUuid uuid of the checkpoint to get
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getCheckpoint(std::string &result,
            const std::string &checkpointUuid,
            Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/checkpoint";
    const std::string vars = "uuid=" + checkpointUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get checkpoint with UUID '" + checkpointUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible checkpoint on shiori
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listCheckpoint(std::string &result,
             Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/checkpoint/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list checkpoints");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a checkpoint
 *
 * @param result reference for response-message
 * @param checkpointUuid uuid of the checkpoint to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteCheckpoint(std::string &result,
               const std::string &checkpointUuid,
               Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/checkpoint";
    const std::string vars = "uuid=" + checkpointUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete checkpoint with UUID '" + checkpointUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace Hanami
