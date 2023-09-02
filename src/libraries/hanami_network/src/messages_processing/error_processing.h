/**
 * @file        error_processing.h
 *
 * @brief       send and handle messages of error-type
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_ERROR_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_ERROR_PROCESSING_H

#include <message_definitions.h>
#include <handler/session_handler.h>
#include <multiblock_io.h>

#include <abstract_socket.h>
#include <hanami_common/buffer/ring_buffer.h>

#include <hanami_network/session_controller.h>
#include <hanami_network/session.h>

#include <hanami_common/logger.h>

namespace Kitsunemimi::Sakura
{

/**
 * @brief send error-message to the other side
 *
 * @param session pointer to the session
 * @param errorCode error-code enum to automatic identify the error-message by code
 * @param message human readable error-message for log-output
 */
inline bool
send_ErrorMessage(Session* session,
                  const uint8_t errorCode,
                  const std::string &errorMessage,
                  ErrorContainer &error)
{
    LOG_DEBUG("SEND error message");

    switch(errorCode)
    {
        //------------------------------------------------------------------------------------------
        case Session::errorCodes::FALSE_VERSION:
        {
            Error_FalseVersion_Message message;

            // fill message
            message.commonHeader.sessionId = session->sessionId();
            message.commonHeader.messageId = session->increaseMessageIdCounter();
            message.messageSize = errorMessage.size();

            // check and copy message-content
            if(message.messageSize > MAX_SINGLE_MESSAGE_SIZE - 1) {
                message.messageSize = MAX_SINGLE_MESSAGE_SIZE - 1;
            }
            strncpy(message.message, errorMessage.c_str(),  message.messageSize);

            // send
            return session->sendMessage(message, error);
        }
        //------------------------------------------------------------------------------------------
        case Session::errorCodes::UNKNOWN_SESSION:
        {
            Error_UnknownSession_Message message;

            // fill message
            message.commonHeader.sessionId = session->sessionId();
            message.commonHeader.messageId = session->increaseMessageIdCounter();
            message.messageSize = errorMessage.size();

            // check and copy message-content
            if(message.messageSize > MAX_SINGLE_MESSAGE_SIZE - 1) {
                message.messageSize = MAX_SINGLE_MESSAGE_SIZE - 1;
            }
            strncpy(message.message,
                    errorMessage.c_str(),
                    message.messageSize);

            // send
            return session->sendMessage(message, error);
        }
        //------------------------------------------------------------------------------------------
        case Session::errorCodes::INVALID_MESSAGE_SIZE:
        {
            Error_InvalidMessage_Message message;

            // fill message
            message.commonHeader.sessionId = session->sessionId();
            message.commonHeader.messageId = session->increaseMessageIdCounter();
            message.messageSize = errorMessage.size();

            // check and copy message-content
            if(message.messageSize > MAX_SINGLE_MESSAGE_SIZE - 1) {
                message.messageSize = MAX_SINGLE_MESSAGE_SIZE - 1;
            }
            strncpy(message.message,
                    errorMessage.c_str(),
                    message.messageSize);

            // send
            return session->sendMessage(message, error);
        }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }

    return true;
}

/**
 * @brief process messages of error-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_Error_Type(Session* session,
                   const CommonMessageHeader* header,
                   const void* rawMessage)
{
    // release session for the case, that the session is actually still in creating state. If this
    // lock is not release, it blocks for eterity.
    session->m_initState = -1;

    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case ERROR_FALSE_VERSION_SUBTYPE:
            {
                const Error_FalseVersion_Message* message =
                    static_cast<const Error_FalseVersion_Message*>(rawMessage);
                session->m_processError(session,
                                        Session::errorCodes::FALSE_VERSION,
                                        std::string(message->message, message->messageSize));
                break;
            }
        //------------------------------------------------------------------------------------------
        case ERROR_UNKNOWN_SESSION_SUBTYPE:
            {
                const Error_UnknownSession_Message* message =
                    static_cast<const Error_UnknownSession_Message*>(rawMessage);
                session->m_processError(session,
                                        Session::errorCodes::UNKNOWN_SESSION,
                                        std::string(message->message, message->messageSize));
                break;
            }
        //------------------------------------------------------------------------------------------
        case ERROR_INVALID_MESSAGE_SUBTYPE:
            {
                const Error_InvalidMessage_Message* message =
                    static_cast<const Error_InvalidMessage_Message*>(rawMessage);
                session->m_processError(session,
                                        Session::errorCodes::INVALID_MESSAGE_SIZE,
                                        std::string(message->message, message->messageSize));
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_ERROR_PROCESSING_H
