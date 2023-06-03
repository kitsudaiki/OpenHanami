/**
 * @file        snapshot.cpp
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

#include <libHanamiAiSdk/snapshot.h>
#include <common/http_client.h>

namespace HanamiAI
{

/**
 * @brief get information of a snapshot from shiori
 *
 * @param result reference for response-message
 * @param snapshotUuid uuid of the snapshot to get
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getSnapshot(std::string &result,
            const std::string &snapshotUuid,
            Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster_snapshot";
    const std::string vars = "uuid=" + snapshotUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get snapshot with UUID '" + snapshotUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible snapshot on shiori
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listSnapshot(std::string &result,
             Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster_snapshot/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list snapshots");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a snapshot
 *
 * @param result reference for response-message
 * @param snapshotUuid uuid of the snapshot to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteSnapshot(std::string &result,
               const std::string &snapshotUuid,
               Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster_snapshot";
    const std::string vars = "uuid=" + snapshotUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete snapshot with UUID '" + snapshotUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace HanamiAI
