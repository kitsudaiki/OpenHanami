/**
 * @file        task.cpp
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

#include <libHanamiAiSdk/task.h>
#include <common/http_client.h>

namespace HanamiAI
{

/**
 * @brief create a new learn-task
 *
 * @param result reference for response-message
 * @param name name of the new task
 * @param type type of the new task (learn or request)
 * @param clusterUuid uuid of the cluster, which should execute the task
 * @param dataSetUuid uuid of the data-set-file on server
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
createTask(std::string &result,
           const std::string &name,
           const std::string &type,
           const std::string &clusterUuid,
           const std::string &dataSetUuid,
           Kitsunemimi::ErrorContainer &error)
{
    // precheck task-type
    if(type != "learn"
            && type != "request")
    {
        error.addMeesage("Unknow task-type '" + type + "'");
        return false;
    }

    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/task";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""
                                 + name
                                 + "\",\"type\":\""
                                 + type
                                 + "\",\"cluster_uuid\":\""
                                 + clusterUuid
                                 + "\",\"data_set_uuid\":\""
                                 + dataSetUuid
                                 + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to start task on cluster with UUID '"
                         + clusterUuid
                         + "' and dataset with UUID '"
                         + dataSetUuid
                         + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get task-information
 *
 * @param result reference for response-message
 * @param taskUuid uuid of the requested task
 * @param clusterUuid uuid of the cluster, where the task belongs to
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getTask(std::string &result,
        const std::string &taskUuid,
        const std::string &clusterUuid,
        Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/task";
    std::string vars = "uuid=" + taskUuid + "&cluster_uuid=" + clusterUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get task with UUID '"
                         + taskUuid
                         + "' of cluster with UUID '"
                         + clusterUuid
                         + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible tasks on hanami
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster, which tasks should be listed
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listTask(std::string &result,
         const std::string &clusterUuid,
         Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/task/all?cluster_uuid=" + clusterUuid;

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list tasks");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete or abort a task from hanami
 *
 * @param result reference for response-message
 * @param taskUuid uuid of the task, which should be deleted
 * @param clusterUuid uuid of the cluster, where the task belongs to
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteTask(std::string &result,
           const std::string &taskUuid,
           const std::string &clusterUuid,
           Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/task";
    const std::string vars = "uuid=" + taskUuid + "&clusterUuid=" + clusterUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete task with UUID '" + taskUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace HanamiAI
