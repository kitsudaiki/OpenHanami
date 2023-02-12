/**
 * @file        callbacks.h
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

#ifndef KITSUNEMIMI_HANAMI_MESSAGING_CALLBACKS_H
#define KITSUNEMIMI_HANAMI_MESSAGING_CALLBACKS_H

#include <message_handling/messaging_event.h>
#include <message_handling/message_definitions.h>
#include <message_handling/messaging_event_queue.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiSakuraNetwork/session_controller.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiJson/json_item.h>

namespace Kitsunemimi::Hanami
{

/**
 * @brief process incoming standalone-message
 *
 * @param session pointer to the session, where the message belongs to
 * @param blockerId blocker-id for the response
 * @param data data-buffer with plain message
 */
void
standaloneDataCallback(void*,
                       Kitsunemimi::Sakura::Session* session,
                       const uint64_t blockerId,
                       Kitsunemimi::DataBuffer* data)
{
    LOG_DEBUG("receive sakura-message");
    if(data->usedBufferSize == 0)
    {
        LOG_WARNING("received empty message");
        delete data;
        return;
    }

    const uint8_t type = static_cast<const uint8_t*>(data->data)[0];

    //==============================================================================================
    if(type == SAKURA_TRIGGER_MESSAGE)
    {
        // prepare message
        const SakuraTriggerHeader* header = static_cast<const SakuraTriggerHeader*>(data->data);
        const char* message = static_cast<const char*>(data->data);

        // get id
        uint32_t pos = sizeof (SakuraTriggerHeader);
        const std::string id(&message[pos], header->idSize);

        // get input-values
        pos += header->idSize;
        const std::string inputValues(&message[pos], header->inputValuesSize);

        LOG_DEBUG("receive sakura-trigger-message for id: " + id
                  + "\n  and input-values: " + inputValues);

        // create new event and place it within the event-queue
        MessagingEvent* event = new MessagingEvent(header->requestType,
                                                   id,
                                                   inputValues,
                                                   session,
                                                   blockerId);

        MessagingEventQueue::getInstance()->addEventToQueue(event);
    }
    //==============================================================================================
    if(type == SAKURA_GENERIC_MESSAGE)
    {
        const SakuraGenericHeader* header = static_cast<const SakuraGenericHeader*>(data->data);
        char* msgBody = &static_cast<char*>(data->data)[sizeof(SakuraGenericHeader)];
        HanamiMessaging::getInstance()->processGenericRequest(session,
                                                              header->subType,
                                                              msgBody,
                                                              header->size,
                                                              blockerId);
    }
    //==============================================================================================
    // TODO: error when unknown

    delete data;
}

/**
 * @brief error-callback
 */
void
errorCallback(Sakura::Session* session,
              const uint8_t,
              const std::string message)
{
    Kitsunemimi::ErrorContainer error;
    error.addMeesage(message);
    LOG_ERROR(error);

    const std::string identifier = session->m_sessionIdentifier;
    error.addMeesage("try to close session after error with identifier: '" + identifier + "'");

    // close-session
    if(session->isClientSide()) {
        HanamiMessaging::getInstance()->closeClient(identifier, error);
    } else {
        HanamiMessaging::getInstance()->removeInternalClient(identifier);
    }

    LOG_ERROR(error);
}

/**
 * @brief callback for new sessions
 *
 * @param session pointer to session
 * @param identifier identifier of the incoming session
 */
void
sessionCreateCallback(Kitsunemimi::Sakura::Session* session,
                      const std::string identifier)
{
    // set callback for incoming standalone-messages for trigger sakura-files
    session->setRequestCallback(nullptr, &standaloneDataCallback);
    session->setStreamCallback(HanamiMessaging::getInstance()->streamReceiver,
                               HanamiMessaging::getInstance()->processStreamData);

    // callback was triggered on server-side, place new session into central list
    if(session->isClientSide() == false) {
        HanamiMessaging::getInstance()->addInternalClient(identifier, session);
    }
}

/**
 * @brief callback for closing sessions
 *
 * @param identifier identifier of the incoming session
 */
void
sessionCloseCallback(Kitsunemimi::Sakura::Session* session,
                      const std::string identifier)
{
    Kitsunemimi::ErrorContainer error;
    LOG_INFO("try to close session with identifier: '" + identifier + "'");

    // close-session
    if(session->isClientSide() == false) {
        HanamiMessaging::getInstance()->removeInternalClient(identifier);
    }
}

}

#endif // KITSUNEMIMI_HANAMI_MESSAGING_CALLBACKS_H
