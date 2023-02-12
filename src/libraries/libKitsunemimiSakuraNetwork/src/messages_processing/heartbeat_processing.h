/**
 * @file        heartbeat_processing.h
 *
 * @brief       send and handle messages of heartbeat-type
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_HEARTBEAT_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_HEARTBEAT_PROCESSING_H

#include <message_definitions.h>
#include <handler/session_handler.h>
#include <multiblock_io.h>

#include <libKitsunemimiNetwork/abstract_socket.h>
#include <libKitsunemimiCommon/buffer/ring_buffer.h>

#include <libKitsunemimiSakuraNetwork/session_controller.h>
#include <libKitsunemimiSakuraNetwork/session.h>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi::Sakura
{

/**
 * @brief send the initial message
 *
 * @param session pointer to the session
 */
inline bool
send_Heartbeat_Start(Session* session,
                     ErrorContainer &error)
{
    Heartbeat_Start_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = session->increaseMessageIdCounter();

    return session->sendMessage(message, error);
}

/**
 * @brief send reply-message
 *
 * @param session pointer to the session
 * @param id of the message of the initial heartbeat
 */
inline bool
send_Heartbeat_Reply(Session* session,
                     const uint32_t messageId,
                     ErrorContainer &error)
{
    Heartbeat_Reply_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = messageId;

    return session->sendMessage(message, error);
}

/**
 * @brief handle incoming heartbeats by sending a reply mesage
 *
 * @param session pointer to the session
 * @param message incoming message
 */
inline bool
process_Heartbeat_Start(Session* session,
                        const Heartbeat_Start_Message* message)
{
    return send_Heartbeat_Reply(session, message->commonHeader.messageId, session->sessionError);
}

/**
 * @brief handle the reply-message, but do nothing here, because it is only important, that the
 *        message is arrived to be handled by the timer-thread
 */
inline void
process_Heartbeat_Reply(Session*, const Heartbeat_Reply_Message*)
{
    return;
}

/**
 * @brief process messages of heartbeat-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_Heartbeat_Type(Session* session,
                       const CommonMessageHeader* header,
                       const void* rawMessage)
{
    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case HEARTBEAT_START_SUBTYPE:
            {
                const Heartbeat_Start_Message* message =
                    static_cast<const Heartbeat_Start_Message*>(rawMessage);
                process_Heartbeat_Start(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        case HEARTBEAT_REPLY_SUBTYPE:
            {
                const Heartbeat_Reply_Message* message =
                    static_cast<const Heartbeat_Reply_Message*>(rawMessage);
                process_Heartbeat_Reply(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_HEARTBEAT_PROCESSING_H
