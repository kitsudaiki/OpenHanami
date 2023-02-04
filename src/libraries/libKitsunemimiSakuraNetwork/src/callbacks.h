/**
 * @file       callbacks.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_CALLBACKS_H
#define KITSUNEMIMI_SAKURA_NETWORK_CALLBACKS_H

#include <libKitsunemimiNetwork/abstract_socket.h>
#include <libKitsunemimiCommon/buffer/ring_buffer.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

#include <libKitsunemimiSakuraNetwork/session_controller.h>

#include <messages_processing/session_processing.h>
#include <messages_processing/heartbeat_processing.h>
#include <messages_processing/error_processing.h>
#include <messages_processing/stream_data_processing.h>
#include <messages_processing/multiblock_data_processing.h>
#include <messages_processing/singleblock_data_processing.h>

namespace Kitsunemimi
{
namespace Sakura
{

/**
 * @brief hexlify an object into a string
 *
 * @param object pointer to the object, which should be hexlified
 * @param size size of the object
 *
 * @return hex-version of the input-object as string
 */
void
hexlify(std::string &outputString,
        const void* object,
        const uint64_t size)
{
    const uint8_t* bytestream = static_cast<const uint8_t*>(object);
    std::stringstream stream;

    // iterate over all bytes of the object
    for(uint64_t i = size; i != 0; i--)
    {
        // special rul to fix the length
        if(bytestream[i - 1] < 16) {
            stream << "0";
        }
        stream << std::hex << static_cast<int>(bytestream[i - 1]);
    }

    outputString = stream.str();
}

/**
 * @brief createHeaderError
 * @param header
 */
void
createHeaderError(const std::string &message,
                  const CommonMessageHeader* header)
{
    std::string headerContent = "";
    hexlify(headerContent, &header, sizeof(header));

    ErrorContainer error;
    error.addMeesage(message);
    error.addSolution(" check header: " + headerContent);
    LOG_ERROR(error);
}

/**
 * process incoming data
 *
 * @param target void-pointer to the session, which had received the message
 * @param recvBuffer data-buffer with the incoming data
 *
 * @return number of bytes, which were taken from the buffer
 */
inline uint64_t
processMessage(void* target,
               RingBuffer* recvBuffer)
{
    // gsession, which is related to the message
    Session* session = static_cast<Session*>(target);

    // et header of message and check if header was complete within the buffer
    const CommonMessageHeader* header = getObject_RingBuffer<CommonMessageHeader>(*recvBuffer);
    if(header == nullptr) {
        return 0;
    }

    // check for correct protocol
    if(header->protocolIdentifier != PROTOCOL_IDENTIFIER)
    {
        createHeaderError("invalid incoming protocol", header);

        // close session, because its an invalid incoming protocol
        ErrorContainer error;
        session->closeSession(error);
        LOG_ERROR(error);
        return 0;
    }

    // check version in header
    if(header->version != 0x1)
    {
        ErrorContainer error;
        send_ErrorMessage(session, Session::errorCodes::FALSE_VERSION, "", error);
        LOG_ERROR(error);
        createHeaderError("false message-version", header);
        return 0;
    }

    // get complete message from the ringbuffer, if enough data are available
    const void* rawMessage = getDataPointer_RingBuffer(*recvBuffer, header->totalMessageSize);
    if(rawMessage == nullptr) {
        return 0;
    }

    // check delimiter of the message
    //    devide by 4, because uint32_t has 4 bytes sizede
    const uint64_t endPosition = (header->totalMessageSize / 4) - 1;
    const uint32_t* end = static_cast<const uint32_t*>(rawMessage) + endPosition;
    if(*end != MESSAGE_DELIMITER)
    {
        ErrorContainer error;
        send_ErrorMessage(session, Session::errorCodes::FALSE_VERSION, "", error);
        LOG_ERROR(error);
        createHeaderError("delimiter does not match", header);
        assert(false);
        return 0;
    }

    // remove from reply-handler if message is reply
    if(header->flags & 0x2) {
        SessionHandler::m_replyHandler->removeMessage(header->sessionId, header->messageId);
    }

    // process message by type
    switch(header->type)
    {
        case STREAM_DATA_TYPE:
            process_Stream_Data_Type(session, header, rawMessage);
            break;
        case SINGLEBLOCK_DATA_TYPE:
            process_SingleBlock_Data_Type(session, header, rawMessage);
            break;
        case MULTIBLOCK_DATA_TYPE:
            process_MultiBlock_Data_Type(session, header, rawMessage);
            break;
        case SESSION_TYPE:
            process_Session_Type(session, header, rawMessage);
            break;
        case HEARTBEAT_TYPE:
            process_Heartbeat_Type(session, header, rawMessage);
            break;
        case ERROR_TYPE:
            process_Error_Type(session, header, rawMessage);
            break;
        default:
            // TODO: handle invalid case
            return 0;
    }

    return header->totalMessageSize;
}

/**
 * process incoming data
 *
 * @param target void-pointer to the session, which had received the message
 * @param recvBuffer data-buffer with the incoming data
 *
 * @return number of bytes, which were taken from the buffer
 */
uint64_t
processMessage_callback(void* target,
                        RingBuffer* recvBuffer,
                        AbstractSocket*)
{
    return processMessage(target, recvBuffer);
}

/**
 * @brief triggered for a new incoming connection
 *
 * @param socket socket for the new session
 */
void
processConnection_Callback(void*,
                           AbstractSocket* socket)
{
    Session* newSession = new Session(socket);
    socket->setMessageCallback(newSession, &processMessage_callback);
    socket->startThread();
}

} // namespace Sakura
} // namespace Kitsunemimi

#endif // KITSUNEMIMI_SAKURA_NETWORK_CALLBACKS_H
