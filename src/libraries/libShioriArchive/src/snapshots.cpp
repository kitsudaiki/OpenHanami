/**
 * @file        snapshots.cpp
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

#include <libShioriArchive/snapshots.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <../../libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3.pb.h>
#include <../../libKitsunemimiHanamiMessages/message_sub_types.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::HanamiMessagingClient;
using Kitsunemimi::Hanami::SupportedComponents;

namespace Shiori
{

/**
 * @brief get data of a snapshot from shiori
 *
 * @param location file-location of the snapshot within shiori
 * @param error reference for error-output
 *
 * @return pointer to buffer with the data of the snapshot, if successful, else nullptr
 */
Kitsunemimi::DataBuffer*
getSnapshotData(const std::string &location,
                Kitsunemimi::ErrorContainer &error)
{
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;

    // create message
    ClusterSnapshotPull_Message msg;
    msg.set_location(location);

    // serialize message
    uint8_t buffer[96*1024];
    const uint64_t msgSize = msg.ByteSizeLong();
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize message");
        return nullptr;
    }

    // send message
    return client->sendGenericRequest(SHIORI_CLUSTER_SNAPSHOT_PULL_MESSAGE_TYPE,
                                      buffer,
                                      msgSize,
                                      error);
}

/**
 * @brief get information of a specific snapshot from shiori
 *
 * @param result reference for the output of the resulting json
 * @param snapshotUuid uuid of the requested snapshot
 * @param token access-token for shiori
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getSnapshotInformation(Kitsunemimi::JsonItem &result,
                       const std::string &snapshotUuid,
                       const std::string &token,
                       Kitsunemimi::ErrorContainer &error)
{
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr) {
        return false;
    }

    Kitsunemimi::Hanami::ResponseMessage response;

    // create request for remote-calls
    Kitsunemimi::Hanami::RequestMessage request;
    request.id = "v1/cluster_snapshot";
    request.httpType = Kitsunemimi::Hanami::GET_TYPE;
    request.inputValues = "{\"uuid\" : \"" + snapshotUuid + "\","
                          "\"token\":\"" + token + "\"}";

    // send request to the target
    if(client->triggerSakuraFile(response, request, error) == false) {
        return false;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage(response.responseContent);
        return false;
    }

    // parse result
    if(result.parse(response.responseContent, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief initialize the transfer of the cluster-snapshot to shiori
 *
 * @param fileUuid uuid of the temporary file of the snapshot in shiori
 * @param snapshotUuid uuid of the new snapshot, which should be the same like the task-uuid
 * @param snapshotName name of the new snapshot
 * @param userId id of the user who owns the snapshot
 * @param projectId id of the project in with the snapshot was created
 * @param totalSize total size of the snapshot
 * @param headerMessage header-message with meta-information of the snapshot
 * @param token access-token for shiori
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
runSnapshotInitProcess(std::string &fileUuid,
                       const std::string &snapshotUuid,
                       const std::string &snapshotName,
                       const std::string &userId,
                       const std::string &projectId,
                       const uint64_t totalSize,
                       const std::string &headerMessage,
                       const std::string &token,
                       Kitsunemimi::ErrorContainer &error)
{
    // get internal client for interaction with shiori
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client to shiori");
        error.addSolution("Check if shiori is correctly configured");
        return false;
    }

    // create request
    Kitsunemimi::Hanami::RequestMessage requestMsg;
    requestMsg.id = "v1/cluster_snapshot";
    requestMsg.httpType = Kitsunemimi::Hanami::HttpRequestType::POST_TYPE;
    requestMsg.inputValues = "";
    requestMsg.inputValues.append("{\"user_id\":\"");
    requestMsg.inputValues.append(userId);
    requestMsg.inputValues.append("\",\"token\":\"");
    requestMsg.inputValues.append(token);
    requestMsg.inputValues.append("\",\"uuid\":\"");
    requestMsg.inputValues.append(snapshotUuid);
    requestMsg.inputValues.append("\",\"header\":");
    requestMsg.inputValues.append(headerMessage);
    requestMsg.inputValues.append(",\"project_id\":\"");
    requestMsg.inputValues.append(projectId);
    requestMsg.inputValues.append("\",\"name\":\"");
    requestMsg.inputValues.append(snapshotName);
    requestMsg.inputValues.append("\",\"input_data_size\":");
    requestMsg.inputValues.append(std::to_string(totalSize));
    requestMsg.inputValues.append("}");

    // trigger initializing of snapshot
    Kitsunemimi::Hanami::ResponseMessage response;
    if(client->triggerSakuraFile(response, requestMsg, error) == false)
    {
        error.addMeesage("Failed to trigger blossom in shiori to initialize "
                         "the transfer of a cluster");
        return false;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage(response.responseContent);
        error.addMeesage("Failed to trigger blossom in shiori to initialize "
                         "the transfer of a cluster");
        return false;
    }

    // process response
    LOG_DEBUG("Response from initializing cluster-snapshot: " + response.responseContent);
    Kitsunemimi::JsonItem parsedResponse;
    if(parsedResponse.parse(response.responseContent, error) == false)
    {
        error.addMeesage("Failed to parse reponse from shiori for the initializing "
                         "of the snapshot-transfer");
        return false;
    }
    fileUuid = parsedResponse.get("uuid_input_file").getString();

    return true;
}

/**
 * @brief send data of the snapshot to shiori
 *
 * @param data buffer with data to send
 * @param targetPos byte-position within the snapshot where the data belongs to
 * @param uuid uuid of the snapshot
 * @param fileUuid uuid of the temporary file in shiori for identification
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendData(const Kitsunemimi::DataBuffer* data,
         uint64_t &targetPos,
         const std::string &uuid,
         const std::string &fileUuid,
         Kitsunemimi::ErrorContainer &error)
{
    // get internal client for interaction with shiori
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client to shiori");
        error.addSolution("Check if shiori is correctly configured");
        return false;
    }

    const uint64_t dataSize = data->usedBufferSize;
    const uint8_t* u8Data = static_cast<const uint8_t*>(data->data);

    uint8_t sendBuffer[128*1024];
    uint64_t i = 0;
    uint64_t pos = 0;

    // prepare buffer
    uint64_t segmentSize = 96 * 1024;

    do
    {
        pos = i + targetPos;

        FileUpload_Message message;
        message.set_fileuuid(fileUuid);
        message.set_datasetuuid(uuid);
        message.set_type(UploadDataType::CLUSTER_SNAPSHOT_TYPE);
        message.set_islast(false);

        // check the size for the last segment
        segmentSize = 96 * 1024;
        if(dataSize - i < segmentSize)
        {
            segmentSize = dataSize - i;
            message.set_islast(true);
        }

        // read segment of the local file
        message.set_position(pos);
        message.set_data(static_cast<const void*>(&u8Data[i]), segmentSize);

        const uint64_t msgSize = message.ByteSizeLong();
        if(message.SerializeToArray(sendBuffer, msgSize) == false)
        {
            error.addMeesage("Failed to serialize learn-message");
            return false;
        }

        // send segment
        if(client->sendStreamMessage(sendBuffer, msgSize, false, error) == false)
        {
            error.addMeesage("Failed to send part with position '"
                             + std::to_string(i)
                             + "' to shiori");
            return false;
        }

        i += segmentSize;
    }
    while(i < dataSize);

    targetPos += i;

    return true;
}

/**
 * @brief finalize the transfer of the snapshot to shiori
 *
 * @param snapshotUuid uuid of the snapshot to finalize
 * @param fileUuid uuid of the temporary file of the snapshot in shiori
 * @param token access-token for shiori
 * @param userId id of the user who owns the snapshot
 * @param projectId id of the project in with the snapshot was created
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
runSnapshotFinalizeProcess(const std::string &snapshotUuid,
                           const std::string &fileUuid,
                           const std::string &token,
                           const std::string &userId,
                           const std::string &projectId,
                           Kitsunemimi::ErrorContainer &error)
{
    // get internal client for interaction with shiori
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client to shiori");
        error.addSolution("Check if shiori is correctly configured");
        return false;
    }

    // create request
    Kitsunemimi::Hanami::RequestMessage requestMsg;
    requestMsg.id = "v1/cluster_snapshot";
    requestMsg.httpType = Kitsunemimi::Hanami::HttpRequestType::PUT_TYPE;
    requestMsg.inputValues = "";
    requestMsg.inputValues.append("{\"user_id\":\"");
    requestMsg.inputValues.append(userId);
    requestMsg.inputValues.append("\",\"token\":\"");
    requestMsg.inputValues.append(token);
    requestMsg.inputValues.append("\",\"project_id\":\"");
    requestMsg.inputValues.append(projectId);
    requestMsg.inputValues.append("\",\"uuid\":\"");
    requestMsg.inputValues.append(snapshotUuid);
    requestMsg.inputValues.append("\",\"uuid_input_file\":\"");
    requestMsg.inputValues.append(fileUuid);
    requestMsg.inputValues.append("\"}");

    // trigger finalizing of snapshot
    Kitsunemimi::Hanami::ResponseMessage response;
    if(client->triggerSakuraFile(response, requestMsg, error) == false)
    {
        error.addMeesage("Failed to trigger blossom in shiori to finalize "
                         "the transfer of a cluster");
        return false;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage(response.responseContent);
        error.addMeesage("Failed to trigger blossom in shiori to finalize "
                         "the transfer of a cluster");
        return false;
    }

    // process response
    LOG_DEBUG("Response from finalizing cluster-snapshot: " + response.responseContent);
    Kitsunemimi::JsonItem parsedResponse;
    if(parsedResponse.parse(response.responseContent, error) == false)
    {
        error.addMeesage("Failed to parse reponse from shiori for the finalizeing "
                         "of the snapshot-transfer");
        return false;
    }

    return true;
}

}
