/**
 * @file        other.h
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

#include <libShioriArchive/other.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCrypto/common.h>

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiCommon/structs.h>
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
 * @brief send list with request-results to shiori
 *
 * @param uuid uuid of the request-task
 * @param name name of the request-task
 * @param results data-array with results
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendResults(const std::string &uuid,
            const std::string &name,
            const std::string &userId,
            const std::string &projectId,
            const Kitsunemimi::DataArray &results,
            Kitsunemimi::ErrorContainer &error)
{
    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client for connection to shiori");
        return false;
    }

    // create message
    ResultPush_Message msg;
    msg.set_uuid(uuid);
    msg.set_name(name);
    msg.set_userid(userId);
    msg.set_projectid(projectId);
    msg.set_results(results.toString());

    // serialize message
    const uint64_t msgSize = msg.ByteSizeLong();
    uint8_t* buffer = new uint8_t[msgSize];
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize error-message to shiori");
        delete[] buffer;
        return false;
    }

    // send message
    Kitsunemimi::DataBuffer* ret = client->sendGenericRequest(SHIORI_RESULT_PUSH_MESSAGE_TYPE,
                                                              buffer,
                                                              msgSize,
                                                              error);
    if(ret == nullptr)
    {
        error.addMeesage("Failed to send result-message to shiori");
        delete[] buffer;
        return false;
    }

    delete ret;
    delete[] buffer;

    return true;
}

/**
 * @brief send error-message to shiori
 *
 * @param userId id of the user where the error belongs to
 * @param errorMessage error-message to send to shiori
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendErrorMessage(const std::string &userId,
                 const std::string &errorMessage,
                 Kitsunemimi::ErrorContainer &error)
{
    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client for connection to shiori");
        return false;
    }

    // create message
    ErrorLog_Message msg;
    msg.set_userid(userId);
    msg.set_errormsg(errorMessage);

    // serialize message
    uint8_t buffer[96*1024];
    const uint64_t msgSize = msg.ByteSizeLong();
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize error-message to shiori");
        return false;
    }

    // send message
    if(client->sendGenericMessage(SHIORI_ERROR_LOG_MESSAGE_TYPE, buffer, msgSize, error) == false)
    {
        error.addMeesage("Failed to send error-message to shiori");
        return false;
    }

    return true;
}

/**
 * @brief send audit-log-entry to shiori
 *
 * @param targetComponent accessed component
 * @param targetEndpoint accessed endpoint
 * @param userId user-id who made the request to the endpoint
 * @param requestType http-type of the request
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendAuditMessage(const std::string &targetComponent,
                 const std::string &targetEndpoint,
                 const std::string &userId,
                 const Kitsunemimi::Hanami::HttpRequestType requestType,
                 Kitsunemimi::ErrorContainer &error)
{
    // check if shiori is supported
    if(SupportedComponents::getInstance()->support[Kitsunemimi::Hanami::SHIORI] == false) {
        return false;
    }

    // convert http-type into string
    std::string httpType = "GET";
    if(requestType == Kitsunemimi::Hanami::DELETE_TYPE) {
        httpType = "DELETE";
    }
    if(requestType == Kitsunemimi::Hanami::GET_TYPE) {
        httpType = "GET";
    }
    if(requestType == Kitsunemimi::Hanami::HEAD_TYPE) {
        httpType = "HEAD";
    }
    if(requestType == Kitsunemimi::Hanami::POST_TYPE) {
        httpType = "POST";
    }
    if(requestType == Kitsunemimi::Hanami::PUT_TYPE) {
        httpType = "PUT";
    }

    LOG_DEBUG("process uri: \'" + targetEndpoint + "\' with type '" + httpType + "'");

    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client for connection to shiori");
        return false;
    }

    // create message
    AuditLog_Message msg;
    msg.set_userid(userId);
    msg.set_type(httpType);
    msg.set_component(targetComponent);
    msg.set_endpoint(targetEndpoint);

    // serialize message
    uint8_t buffer[96*1024];
    const uint64_t msgSize = msg.ByteSizeLong();
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize audit-message to shiori");
        return false;
    }

    // send message
    if(client->sendGenericMessage(SHIORI_AUDIT_LOG_MESSAGE_TYPE, buffer, msgSize, error) == false)
    {
        error.addMeesage("Failed to send audit-message to shiori");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

}
