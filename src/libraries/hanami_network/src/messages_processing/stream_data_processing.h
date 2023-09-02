/**
 * @file       singleblock_data_processing.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_STREAM_DATA_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_STREAM_DATA_PROCESSING_H

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
 * @brief send_Data_Stream_Static
 */
inline bool
send_Data_Stream(Session* session,
                 DataBuffer* data,
                 const bool replyExpected,
                 ErrorContainer &error)
{
    if(data->usedBufferSize < sizeof(CommonMessageFooter) + sizeof(Data_Stream_Header)) {
        return false;
    }

    // bring message-size to a multiple of 8
    const uint32_t size = static_cast<uint32_t>(data->usedBufferSize)
                                                - sizeof(CommonMessageFooter)
                                                - sizeof(Data_Stream_Header);
    const uint32_t totalMessageSize = sizeof(Data_Stream_Header)
                                      + size
                                      + (8-(size % 8)) % 8  // fill up to a multiple of 8
                                      + sizeof(CommonMessageFooter);

    CommonMessageFooter end;
    Data_Stream_Header header;

    // fill message
    header.commonHeader.sessionId = session->sessionId();
    header.commonHeader.messageId = session->increaseMessageIdCounter();
    header.commonHeader.totalMessageSize = totalMessageSize;
    header.commonHeader.payloadSize = size;
    header.commonHeader.flags = static_cast<uint8_t>(replyExpected) * 0x1;

    uint8_t* dataPtr = static_cast<uint8_t*>(data->data);
    // fill buffer to build the complete message
    memcpy(&dataPtr[0], &header, sizeof(Data_Stream_Header));
    memcpy(&dataPtr[(totalMessageSize - sizeof(CommonMessageFooter))],
           &end,
           sizeof(CommonMessageFooter));

    // send
    return session->sendMessage(header.commonHeader, dataPtr, totalMessageSize, error);
}

/**
 * @brief send_Data_Stream_Static
 */
inline bool
send_Data_Stream(Session* session,
                 const void* data,
                 const uint32_t size,
                 const bool replyExpected,
                 ErrorContainer &error)
{
    uint8_t messageBuffer[MESSAGE_CACHE_SIZE];

    // bring message-size to a multiple of 8
    const uint32_t totalMessageSize = sizeof(Data_Stream_Header)
                                      + size
                                      + (8 - (size % 8)) % 8  // fill up to a multiple of 8
                                      + sizeof(CommonMessageFooter);

    CommonMessageFooter end;
    Data_Stream_Header header;

    // fill message
    header.commonHeader.sessionId = session->sessionId();
    header.commonHeader.messageId = session->increaseMessageIdCounter();
    header.commonHeader.totalMessageSize = totalMessageSize;
    header.commonHeader.payloadSize = size;
    header.commonHeader.flags = static_cast<uint8_t>(replyExpected) * 0x1;

    // fill buffer to build the complete message
    memcpy(&messageBuffer[0], &header, sizeof(Data_Stream_Header));
    memcpy(&messageBuffer[sizeof(Data_Stream_Header)], data, size);
    memcpy(&messageBuffer[(totalMessageSize - sizeof(CommonMessageFooter))],
           &end,
           sizeof(CommonMessageFooter));

    // send
    return session->sendMessage(header.commonHeader, messageBuffer, totalMessageSize, error);
}

/**
 * @brief send_Data_Stream_Reply
 */
inline bool
send_Data_Stream_Reply(Session* session,
                       const uint32_t messageId,
                       ErrorContainer &error)
{
    Data_StreamReply_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = messageId;

    return session->sendMessage(message, error);
}

/**
 * @brief process_Data_Stream_Static
 */
inline void
process_Data_Stream(Session* session,
                    const Data_Stream_Header* header,
                    const void* rawMessage)
{
    // get pointer to the beginning of the payload
    const uint8_t* payloadData = static_cast<const uint8_t*>(rawMessage)
                                 + sizeof(Data_Stream_Header);

    // trigger callback
    session->m_processStreamData(session->m_streamReceiver,
                                 session,
                                 static_cast<const void*>(payloadData),
                                 header->commonHeader.payloadSize);

    // send reply if necessary
    if(header->commonHeader.flags & 0x1) {
        send_Data_Stream_Reply(session, header->commonHeader.messageId, session->sessionError);
    }
}

/**
 * @brief process_Data_Stream_Reply
 */
inline void
process_Data_Stream_Reply(Session*,
                          const Data_StreamReply_Message*)
{
    return;
}

/**
 * @brief process messages of stream-message-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_Stream_Data_Type(Session* session,
                         const CommonMessageHeader* header,
                         const void* rawMessage)
{
    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case DATA_STREAM_STATIC_SUBTYPE:
            {
                const Data_Stream_Header* message =
                    static_cast<const Data_Stream_Header*>(rawMessage);
                process_Data_Stream(session, message, rawMessage);
                break;
            }
        //------------------------------------------------------------------------------------------
        case DATA_STREAM_REPLY_SUBTYPE:
            {
                const Data_StreamReply_Message* message =
                    static_cast<const Data_StreamReply_Message*>(rawMessage);
                process_Data_Stream_Reply(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_STREAM_DATA_PROCESSING_H
