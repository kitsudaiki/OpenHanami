/**
 * @file        messaging_event.cpp
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

#include "messaging_event.h"
#include "permission.h"

#include <message_handling/message_definitions.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>
#include <libKitsunemimiHanamiCommon/component_support.h>

#include <libKitsunemimiSakuraNetwork/session.h>

#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCrypto/common.h>

#include <../../libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3.pb.h>
#include <../../libKitsunemimiHanamiMessages/message_sub_types.h>

namespace Kitsunemimi
{
namespace Hanami
{

/**
 * @brief constructor
 *
 * @param targetId id of target to trigger
 * @param inputValues input-values as string
 * @param session pointer to session to send the response back
 * @param blockerId blocker-id for the response
 */
MessagingEvent::MessagingEvent(const HttpRequestType httpType,
                               const std::string &targetId,
                               const std::string &inputValues,
                               Kitsunemimi::Sakura::Session* session,
                               const uint64_t blockerId)
{
    m_httpType = httpType;
    m_targetId = targetId;
    m_inputValues = inputValues;
    m_session = session;
    m_blockerId = blockerId;
}

/**
 * @brief destructor
 */
MessagingEvent::~MessagingEvent() {}

/**
 * @brief send reponse message with the results of the event
 *
 * @param success success-result of the event
 * @param responseType response http-type
 * @param message message to send over the response
 * @param session pointer to session to send the response back
 * @param blockerId blocker-id for the response
 * @param error reference for error-output
 */
void
MessagingEvent::sendResponseMessage(const bool success,
                                    const HttpResponseTypes responseType,
                                    const std::string &message,
                                    Kitsunemimi::Sakura::Session* session,
                                    const uint64_t blockerId,
                                    ErrorContainer &error)
{
    // allocate memory to fill with the response-message
    const uint32_t responseMessageSize = sizeof(ResponseHeader)
                                         + static_cast<uint32_t>(message.size());
    uint8_t* buffer = new uint8_t[responseMessageSize];

    // prepare response-header
    ResponseHeader responseHeader;
    responseHeader.success = success;
    responseHeader.responseType = responseType;
    responseHeader.messageSize =  static_cast<uint32_t>(message.size());

    uint32_t positionCounter = 0;

    // copy header and id
    memcpy(buffer, &responseHeader, sizeof(ResponseHeader));
    positionCounter += sizeof(ResponseHeader);
    memcpy(buffer + positionCounter, message.c_str(), message.size());

    // send reponse over the session
    session->sendResponse(buffer, responseMessageSize, blockerId, error);

    delete[] buffer;
}

/**
 * @brief trigger remote blossom or tree
 *
 * @param resultingItems reference for the result of the trigger
 * @param inputValues input-values for the trigger
 * @param status reference for status-output
 * @param endpoint entpoint-entry to identify the target
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
MessagingEvent::trigger(DataMap &resultingItems,
                        JsonItem &inputValues,
                        Hanami::BlossomStatus &status,
                        const EndpointEntry &endpoint,
                        ErrorContainer &error)
{
    DataMap context;
    HanamiMessaging* controller = HanamiMessaging::getInstance();

    const std::string token = inputValues["token"].getString();
    // token is moved into the context object, so to not break the check of the input-fileds of the
    // blossoms, we have to remove this here again
    // TODO: handle context in a separate field in the messaging
    if(m_targetId != "v1/auth") {
        inputValues.remove("token");
    }

    const bool skipPermission = m_session->m_sessionIdentifier != "torii"
                                || m_targetId != "v1/auth"
                                || m_targetId != "v1/token"
                                || m_targetId != "v1/token/internal";

    // check permission
    if(checkPermission(context, token, status, skipPermission, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE;
        return false;
    }

    const bool ret = controller->triggerBlossom(resultingItems,
                                                endpoint.name,
                                                endpoint.group,
                                                context,
                                                *inputValues.getItemContent()->toMap(),
                                                status,
                                                error);

    // handle error
    if(ret == false)
    {
        LOG_ERROR(error);
        sendErrorMessage(context, inputValues, error.toString());
        return false;
    }

    return true;
}

/**
 * @brief process messageing-event
 *
 * @return true, if event was successful, else false
 */
bool
MessagingEvent::processEvent()
{
    ErrorContainer error;

    // parse json-formated input values
    JsonItem inputValues;
    if(inputValues.parse(m_inputValues, error) == false)
    {
        LOG_ERROR(error);
        sendResponseMessage(false,
                            BAD_REQUEST_RTYPE,
                            error.toString(),
                            m_session,
                            m_blockerId,
                            error);
        return false;
    }

    // get real endpoint
    EndpointEntry entry;
    bool ret = HanamiMessaging::getInstance()->mapEndpoint(entry, m_targetId, m_httpType);
    if(ret == false)
    {
        error.addMeesage("endpoint not found for id "
                         + m_targetId
                         + " and type "
                         + std::to_string(m_httpType));
        LOG_ERROR(error);
        sendResponseMessage(false,
                            NOT_IMPLEMENTED_RTYPE,
                            error.toString(),
                            m_session,
                            m_blockerId,
                            error);
        return false;
    }

    // execute trigger
    Hanami::BlossomStatus status;
    DataMap resultingItems;
    ret = trigger(resultingItems, inputValues, status, entry, error);

    // creating and send reposonse with the result of the event
    const HttpResponseTypes type = static_cast<HttpResponseTypes>(status.statusCode);
    if(ret)
    {
        sendResponseMessage(true,
                            type,
                            resultingItems.toString(),
                            m_session,
                            m_blockerId,
                            error);
    }
    else
    {
        sendResponseMessage(false,
                            type,
                            status.errorMessage,
                            m_session,
                            m_blockerId,
                            error);
    }

    return true;
}

/**
 * @brief send error-message to shiori
 *
 * @param context context context-object to log
 * @param inputValues inputValues input-values of the request to log
 * @param errorMessage error-message to send to shiori
 */
void
MessagingEvent::sendErrorMessage(const DataMap &context,
                                 const JsonItem &inputValues,
                                 const std::string &errorMessage)
{
    // check if shiori is supported
    if(SupportedComponents::getInstance()->support[SHIORI] == false) {
        return;
    }

    // is user-id is not set, the error is send by the generic error-callback anyway and so it
    // doesn't have to be send twice
    const std::string userId = context.getStringByKey("id");
    if(userId == "") {
        return;
    }

    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr) {
        return;
    }

    // create binary for send
    ErrorLog_Message msg;
    msg.set_userid(userId);
    msg.set_context(context.toString(true));
    msg.set_values(inputValues.toString(true));
    msg.set_component(SupportedComponents::getInstance()->localComponent);
    msg.set_errormsg(errorMessage);

    // serialize message
    const uint64_t msgSize = msg.ByteSizeLong();
    uint8_t* buffer = new uint8_t[msgSize];
    if(msg.SerializeToArray(buffer, msgSize) == false) {
        return;
    }

    // send message
    Kitsunemimi::ErrorContainer error;
    const bool ret = client->sendGenericMessage(SHIORI_ERROR_LOG_MESSAGE_TYPE,
                                                buffer,
                                                msgSize,
                                                error);
    delete[] buffer;
    if(ret == false) {
        return;
    }
}

}  // namespace Hanami
}  // namespace Kitsunemimi
