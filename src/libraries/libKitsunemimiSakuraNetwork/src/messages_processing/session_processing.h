/**
 * @file       session_processing.h
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_SESSION_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_SESSION_PROCESSING_H

#include <message_definitions.h>
#include <handler/session_handler.h>
#include <multiblock_io.h>

#include <abstract_socket.h>

#include <libKitsunemimiSakuraNetwork/session_controller.h>
#include <libKitsunemimiSakuraNetwork/session.h>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi::Sakura
{

/**
 * @brief send_Session_Init_Start
 *
 * @param session pointer to the session
 * @param sessionIdentifier custom value, which is sended within the init-message to pre-identify
 *                          the message on server-side
 */
inline bool
send_Session_Init_Start(Session* session,
                        const std::string &sessionIdentifier,
                        ErrorContainer &error)
{
    LOG_DEBUG("SEND session init start");

    Session_Init_Start_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = session->increaseMessageIdCounter();
    message.clientSessionId = session->sessionId();

    message.sessionIdentifierSize = static_cast<uint32_t>(sessionIdentifier.size());
    memcpy(message.sessionIdentifier, sessionIdentifier.c_str(), sessionIdentifier.size());

    return session->sendMessage(message, error);
}

/**
 * @brief send_Session_Init_Reply
 *
 * @param session pointer to the session
 * @param initialSessionId initial id, which was sended by the client
 * @param messageId id of the original incoming message
 * @param completeSessionId completed session-id based on the id of the server and the client
 * @param sessionIdentifier custom value, which is sended within the init-message to pre-identify
 *                          the message on server-side
 */
inline bool
send_Session_Init_Reply(Session* session,
                        const uint32_t initialSessionId,
                        const uint32_t messageId,
                        const uint32_t completeSessionId,
                        const std::string &sessionIdentifier,
                        ErrorContainer &error)
{
    LOG_DEBUG("SEND session init reply");

    Session_Init_Reply_Message message;

    message.commonHeader.sessionId = initialSessionId;
    message.commonHeader.messageId = messageId;
    message.completeSessionId = completeSessionId;
    message.clientSessionId = initialSessionId;

    message.sessionIdentifierSize = static_cast<uint32_t>(sessionIdentifier.size());
    memcpy(message.sessionIdentifier, sessionIdentifier.c_str(), sessionIdentifier.size());

    return session->sendMessage(message, error);
}

/**
 * @brief send_Session_Close_Start
 *
 * @param session pointer to the session
 * @param replyExpected set to true to get a reply-message for the session-close-message
 */
inline bool
send_Session_Close_Start(Session* session,
                         const bool replyExpected,
                         ErrorContainer &error)
{
    LOG_DEBUG("SEND session close start");

    Session_Close_Start_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = session->increaseMessageIdCounter();
    if(replyExpected) {
        message.commonHeader.flags = 0x1;
    }

    return session->sendMessage(message, error);
}

/**
 * @brief send_Session_Close_Reply
 *
 * @param session pointer to the session
 * @param messageId id of the original incoming message
 */
inline bool
send_Session_Close_Reply(Session* session,
                         const uint32_t messageId,
                         ErrorContainer &error)
{
    LOG_DEBUG("SEND session close reply");

    Session_Close_Reply_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = messageId;

    return session->sendMessage(message, error);
}

/**
 * @brief process_Session_Init_Start
 *
 * @param session pointer to the session
 * @param message pointer to the complete message within the message-ring-buffer
 */
inline void
process_Session_Init_Start(Session* session,
                           const Session_Init_Start_Message* message)
{
    LOG_DEBUG("process session init start");

    // get and calculate session-id
    const uint32_t clientSessionId = message->clientSessionId;
    const uint16_t serverSessionId = SessionHandler::m_sessionHandler->increaseSessionIdCounter();
    const uint32_t sessionId = clientSessionId + (serverSessionId * 0x10000);
    const std::string sessionIdentifier(message->sessionIdentifier, message->sessionIdentifierSize);

    // create new session and make it ready
    SessionHandler::m_sessionHandler->addSession(sessionId, session);
    session->connectiSession(sessionId, session->sessionError);
    session->makeSessionReady(sessionId, sessionIdentifier, session->sessionError);

    // send
    send_Session_Init_Reply(session,
                            clientSessionId,
                            message->commonHeader.messageId,
                            sessionId,
                            sessionIdentifier,
                            session->sessionError);
}

/**
 * @brief process_Session_Init_Reply
 *
 * @param session pointer to the session
 * @param message pointer to the complete message within the message-ring-buffer
 */
inline void
process_Session_Init_Reply(Session* session,
                           const Session_Init_Reply_Message* message)
{
    LOG_DEBUG("process session init reply");

    const uint32_t completeSessionId = message->completeSessionId;
    const uint32_t initialId = message->clientSessionId;
    const std::string sessionIdentifier(message->sessionIdentifier, message->sessionIdentifierSize);

    // readd session under the new complete session-id and make session ready
    SessionHandler::m_sessionHandler->removeSession(initialId);
    SessionHandler::m_sessionHandler->addSession(completeSessionId, session);
    // TODO: handle return-value of makeSessionReady
    session->makeSessionReady(completeSessionId, sessionIdentifier, session->sessionError);
}

/**
 * @brief process_Session_Close_Start
 *
 * @param session pointer to the session
 * @param message pointer to the complete message within the message-ring-buffer
 */
inline void
process_Session_Close_Start(Session* session,
                            const Session_Close_Start_Message* message)
{
    LOG_DEBUG("process session close start");

    send_Session_Close_Reply(session, message->commonHeader.messageId, session->sessionError);

    // close session and disconnect session
    SessionHandler::m_sessionHandler->removeSession(message->sessionId);
    session->endSession(session->sessionError);
}

/**
 * @brief process_Session_Close_Reply
 *
 * @param session pointer to the session
 * @param message pointer to the complete message within the message-ring-buffer
 */
inline void
process_Session_Close_Reply(Session* session,
                            const Session_Close_Reply_Message* message)
{
    LOG_DEBUG("process session close reply");

    // disconnect session
    SessionHandler::m_sessionHandler->removeSession(message->sessionId);
    session->disconnectSession(session->sessionError);
}

/**
 * @brief process messages of session-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_Session_Type(Session* session,
                     const CommonMessageHeader* header,
                     const void* rawMessage)
{
    if(DEBUG_MODE) {
        LOG_DEBUG("process session-type");
    }

    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case SESSION_INIT_START_SUBTYPE:
            {
                const Session_Init_Start_Message* message =
                    static_cast<const Session_Init_Start_Message*>(rawMessage);
                process_Session_Init_Start(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        case SESSION_INIT_REPLY_SUBTYPE:
            {
                const Session_Init_Reply_Message* message =
                    static_cast<const Session_Init_Reply_Message*>(rawMessage);
                process_Session_Init_Reply(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        case SESSION_CLOSE_START_SUBTYPE:
            {
                const Session_Close_Start_Message* message =
                    static_cast<const Session_Close_Start_Message*>(rawMessage);
                process_Session_Close_Start(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        case SESSION_CLOSE_REPLY_SUBTYPE:
            {
                const Session_Close_Reply_Message* message =
                    static_cast<const Session_Close_Reply_Message*>(rawMessage);
                process_Session_Close_Reply(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_SESSION_PROCESSING_H
