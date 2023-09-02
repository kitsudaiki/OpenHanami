/**
 * @file       multiblock_data_processing.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_DATA_PROCESSING_H
#define KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_DATA_PROCESSING_H

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
 * @brief send_Data_Multi_Static
 */
inline bool
send_Data_Multi_Static(Session* session,
                       const uint64_t totalSize,
                       const uint64_t multiblockId,
                       const uint32_t totalPartNumber,
                       const uint32_t partId,
                       const void* data,
                       const uint32_t size,
                       ErrorContainer &error,
                       const uint64_t blockerId = 0)
{
    uint8_t messageBuffer[MESSAGE_CACHE_SIZE];

    // bring message-size to a multiple of 8
    const uint32_t totalMessageSize = sizeof(Data_MultiBlock_Header)
                                      + size
                                      + (8 - (size % 8)) % 8  // fill up to a multiple of 8
                                      + sizeof(CommonMessageFooter);

    CommonMessageFooter end;
    Data_MultiBlock_Header header;

    // fill message
    header.commonHeader.sessionId = session->sessionId();
    header.commonHeader.messageId = session->increaseMessageIdCounter();
    header.commonHeader.totalMessageSize = totalMessageSize;
    header.commonHeader.payloadSize = size;
    header.multiblockId = multiblockId;
    header.totalPartNumber = totalPartNumber;
    header.partId = partId;
    header.totalSize = totalSize;

    // set flag to await response-message for blocker-id
    if(blockerId != 0) {
        header.commonHeader.flags |= 0x8;
    }

    // fill buffer to build the complete message
    memcpy(&messageBuffer[0], &header, sizeof(Data_MultiBlock_Header));
    memcpy(&messageBuffer[sizeof(Data_MultiBlock_Header)], data, size);
    memcpy(&messageBuffer[(totalMessageSize - sizeof(CommonMessageFooter))],
           &end,
           sizeof(CommonMessageFooter));

    return session->sendMessage(header.commonHeader, &messageBuffer, totalMessageSize, error);
}

/**
 * @brief send_Data_Multi_Finish
 */
inline bool
send_Data_Multi_Finish(Session* session,
                       const uint64_t multiblockId,
                       const uint64_t blockerId,
                       ErrorContainer &error)
{
    Data_MultiFinish_Message message;

    message.commonHeader.sessionId = session->sessionId();
    message.commonHeader.messageId = session->increaseMessageIdCounter();
    message.multiblockId = multiblockId;
    message.blockerId = blockerId;
    if(blockerId != 0) {
        message.commonHeader.flags |= 0x8;
    }

    return session->sendMessage(message, error);
}

/**
 * @brief process_Data_Multi_Static
 */
inline void
process_Data_Multiblock(Session* session,
                        const Data_MultiBlock_Header* message,
                        const void* rawMessage)
{
    if(message->partId == 0)
    {
        const bool ret = session->m_multiblockIo->createIncomingBuffer(message->multiblockId,
                                                                       message->totalSize);
        if(ret) {
            // TODO: send error
        }
    }

    const uint8_t* payloadData = static_cast<const uint8_t*>(rawMessage)
                                 + sizeof(Data_MultiBlock_Header);
    session->m_multiblockIo->writeIntoIncomingBuffer(message->multiblockId,
                                                     payloadData,
                                                     message->commonHeader.payloadSize);
}

/**
 * @brief process_Data_Multi_Finish
 */
inline void
process_Data_Multi_Finish(Session* session,
                          const Data_MultiFinish_Message* message)
{
    MultiblockIO::MultiblockBuffer buffer =
            session->m_multiblockIo->getIncomingBuffer(message->multiblockId);

    // check if normal standalone-message or if message is response
    if(message->commonHeader.flags & 0x8)
    {
        // release thread, which is related to the blocker-id
        SessionHandler::m_blockerHandler->releaseMessage(message->blockerId, buffer.incomingData);
    }
    else
    {
        // trigger callback
        session->m_processRequestData(session->m_standaloneReceiver,
                                      session,
                                      message->multiblockId,
                                      buffer.incomingData);
    }

    session->m_multiblockIo->removeMultiblockBuffer(message->multiblockId);
}

/**
 * @brief process messages of multiblock-message-type
 *
 * @param session pointer to the session
 * @param header pointer to the common header of the message within the message-ring-buffer
 * @param rawMessage pointer to the raw data of the complete message (header + payload + end)
 */
inline void
process_MultiBlock_Data_Type(Session* session,
                             const CommonMessageHeader* header,
                             const void* rawMessage)
{
    switch(header->subType)
    {
        //------------------------------------------------------------------------------------------
        case DATA_MULTI_STATIC_SUBTYPE:
            {
                const Data_MultiBlock_Header* message =
                    static_cast<const Data_MultiBlock_Header*>(rawMessage);
                process_Data_Multiblock(session, message, rawMessage);
                break;
            }
        //------------------------------------------------------------------------------------------
        case DATA_MULTI_FINISH_SUBTYPE:
            {
                const Data_MultiFinish_Message* message =
                    static_cast<const Data_MultiFinish_Message*>(rawMessage);
                process_Data_Multi_Finish(session, message);
                break;
            }
        //------------------------------------------------------------------------------------------
        default:
            break;
    }
}

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_DATA_PROCESSING_H
