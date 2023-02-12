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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_SINGLE_DATA_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_SINGLE_DATA_PROCESSING_H

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
 * @brief send_Data_SingleBlock
 */
inline bool
send_Data_SingleBlock(Session* session,
                      const uint64_t multiblockId,
                      const void* data,
                      uint32_t size,
                      ErrorContainer &error,
                      const uint64_t blockerId = 0)
{
    uint8_t messageBuffer[MESSAGE_CACHE_SIZE];

    // bring message-size to a multiple of 8
    const uint32_t totalMessageSize = sizeof(Data_SingleBlock_Header)
                                      + size
                                      + (8 - (size % 8)) % 8  // fill up to a multiple of 8
                                      + sizeof(CommonMessageFooter);

    CommonMessageFooter end;
    Data_SingleBlock_Header header;

    // fill message
    header.commonHeader.sessionId = session->sessionId();
    header.commonHeader.messageId = session->increaseMessageIdCounter();
    header.commonHeader.totalMessageSize = totalMessageSize;
    header.commonHeader.payloadSize = size;
    header.blockerId = blockerId;
    header.multiblockId = multiblockId;

    // set flag to await response-message for blocker-id
    if(blockerId != 0) {
        header.commonHeader.flags |= 0x8;
    }

    // fill buffer with all parts of the message
    memcpy(&messageBuffer[0], &header, sizeof(Data_SingleBlock_Header));
    memcpy(&messageBuffer[sizeof(Data_SingleBlock_Header)], data, size);
    memcpy(&messageBuffer[(totalMessageSize - sizeof(CommonMessageFooter))],
           &end,
           sizeof(CommonMessageFooter));

    // send
    return session->sendMessage(header.commonHeader, &messageBuffer, totalMessageSize, error);
}

/**
 * @brief send_Data_SingleBlock_Reply
 */
inline bool
send_Data_SingleBlock_Reply(Session* session,
                            const uint32_t messageId,
                            ErrorContainer &error)
{
    Data_SingleBlockReply_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = messageId;

    return session->sendMessage(message, error);
}

/**
 * @brief process_Data_SingleBlock
 */
inline void
process_Data_SingleBlock(Session* session,
                         const Data_SingleBlock_Header* header,
                         const void* rawMessage)
{
    // prepare buffer for payload
    const uint32_t payloadSize = header->commonHeader.payloadSize;
    DataBuffer* buffer = new DataBuffer(Kitsunemimi::calcBytesToBlocks(payloadSize));

    // get pointer to the beginning of the payload
    const uint8_t* payloadData = static_cast<const uint8_t*>(rawMessage)
                                 + sizeof(Data_SingleBlock_Header);

    // copy messagy-payload into buffer
    addData_DataBuffer(*buffer, payloadData, header->commonHeader.payloadSize);

    // check if normal standalone-message or if message is response
    if(header->commonHeader.flags & 0x8)
    {
        // release thread, which is related to the blocker-id
        SessionHandler::m_blockerHandler->releaseMessage(header->blockerId,
                                                         buffer);
    }
    else
    {
        // trigger callback
        session->m_processRequestData(session->m_standaloneReceiver,
                                      session,
                                      header->multiblockId,
                                      buffer);
    }

    // send reply, if requested
    if(header->commonHeader.flags & 0x1) {
        send_Data_SingleBlock_Reply(session, header->commonHeader.messageId, session->sessionError);
    }
}

/**
 * @brief process_Data_SingleBlock_Reply
 */
inline void
process_Data_SingleBlock_Reply(Session*,
                               const Data_SingleBlockReply_Message*)
{
    return;
}

/**
 * @brief process messages of singleblock-message-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_SingleBlock_Data_Type(Session* session,
                              const CommonMessageHeader* header,
                              const void* rawMessage)
{
    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case DATA_SINGLE_DATA_SUBTYPE:
            {
                const Data_SingleBlock_Header* message =
                    static_cast<const Data_SingleBlock_Header*>(rawMessage);
                process_Data_SingleBlock(session, message, rawMessage);
                break;
            }
        //------------------------------------------------------------------------------------------
        case DATA_SINGLE_REPLY_SUBTYPE:
            {
                const Data_SingleBlockReply_Message* message =
                    static_cast<const Data_SingleBlockReply_Message*>(rawMessage);
                process_Data_SingleBlock_Reply(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_SINGLE_DATA_PROCESSING_H
