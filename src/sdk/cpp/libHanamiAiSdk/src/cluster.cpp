/**
 * @file        cluster.cpp
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

#include <libHanamiAiSdk/cluster.h>
#include <common/http_client.h>
#include <libHanamiAiSdk/common/websocket_client.h>
#include <hanami_crypto/common.h>

namespace HanamiAI
{

/**
 * @brief create a new cluster from a template on hanami
 *
 * @param result reference for response-message
 * @param clusterName name of the new cluster
 * @param clusterTemplate information to build the new cluster
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
createCluster(std::string &result,
              const std::string &clusterName,
              const std::string &clusterTemplate,
              Hanami::ErrorContainer &error)
{
    HanamiRequest* request = HanamiAI::HanamiRequest::getInstance();

    // convert template into base64-string
    std::string clusterTemplateB64;
    Hanami::encodeBase64(clusterTemplateB64,
                              clusterTemplate.c_str(),
                              clusterTemplate.size());

    // create request
    const std::string path = "/control/v1/cluster";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""
                                 + clusterName
                                 + "\",\"template\":\""
                                 + clusterTemplateB64
                                 + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to create cluster");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get information of a cluster from hanami
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to get
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getCluster(std::string &result,
           const std::string &clusterUuid,
           Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster";
    const std::string vars = "uuid=" + clusterUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get cluster with UUID '" + clusterUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible cluster on hanami
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listCluster(std::string &result,
            Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list clusters");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a cluster with all its tasks from hanami
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteCluster(std::string &result,
              const std::string &clusterUuid,
              Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster";
    const std::string vars = "uuid=" + clusterUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete cluster with UUID '" + clusterUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief create a checkpoint of a cluster
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to delete
 * @param checkpointName name of the new checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
saveCluster(std::string &result,
            const std::string &clusterUuid,
            const std::string &checkpointName,
            Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster/save";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""
                                 + checkpointName
                                 + "\",\"cluster_uuid\":\""
                                 + clusterUuid
                                 + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to save cluster with UUID '" + clusterUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief restore cluster from a checkpoint
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to delete
 * @param checkpointUuid uuid of the checkpoint, which should be loaded into the cluster
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
restoreCluster(std::string &result,
               const std::string &clusterUuid,
               const std::string &checkpointUuid,
               Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster/load";
    const std::string vars = "";
    const std::string jsonBody = "{\"checkpoint_uuid\":\""
                                 + checkpointUuid
                                 + "\",\"cluster_uuid\":\""
                                 + clusterUuid
                                 + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to restore checkpoint with UUID '" + checkpointUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief switch cluster to task-mode
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to swtich
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
switchToTaskMode(std::string &result,
                 const std::string &clusterUuid,
                 Hanami::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster/set_mode";
    const std::string vars = "";
    const std::string jsonBody = "{\"new_state\":\"TASK\""
                                 ",\"uuid\":\""
                                 + clusterUuid
                                 + "\"}";

    // send request
    if(request->sendPutRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to swith cluster with UUID '" + clusterUuid + "' to task-mode");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief switch cluster to direct-mode
 *
 * @param result reference for response-message
 * @param clusterUuid uuid of the cluster to swtich
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
WebsocketClient*
switchToDirectMode(std::string &result,
                   const std::string &clusterUuid,
                   Hanami::ErrorContainer &error)
{
    // init websocket-client
    WebsocketClient* wsClient = new WebsocketClient();
    std::string websocketUuid = "";
    const bool ret = wsClient->initClient(websocketUuid,
                                          HanamiRequest::getInstance()->getToken(),
                                          "kyouko",
                                          HanamiRequest::getInstance()->getHost(),
                                          HanamiRequest::getInstance()->getPort(),
                                          clusterUuid,
                                          error);
    if(ret == false)
    {
        error.addMeesage("Failed to init websocket to hanami");
        LOG_ERROR(error);
        delete wsClient;
        return nullptr;
    }

    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/cluster/set_mode";
    const std::string vars = "";
    const std::string jsonBody = "{\"connection_uuid\":\""
                                 + websocketUuid
                                 + "\",\"new_state\":\"DIRECT\""
                                   ",\"uuid\":\""
                                 + clusterUuid
                                 + "\"}";

    // send request
    if(request->sendPutRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to swith cluster with UUID '" + clusterUuid + "' to direct-mode");
        LOG_ERROR(error);
        delete wsClient;
        return nullptr;
    }

    return wsClient;
}

} // namespace HanamiAI
