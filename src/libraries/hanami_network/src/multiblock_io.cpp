/**
 * @file       multiblock_io.cpp
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

#include "multiblock_io.h"

#include <hanami_common/logger.h>
#include <hanami_network/session.h>
#include <messages_processing/multiblock_data_processing.h>

namespace Hanami
{

MultiblockIO::MultiblockIO(Session* session) { m_session = session; }

MultiblockIO::~MultiblockIO()
{
    m_abort = true;
    usleep(10000);

    for (auto const& [id, buffer] : m_incomingBuffer) {
        delete buffer.incomingData;
    }
}

/**
 * @brief send multiblock-message
 *
 * @param data payload of the message to send
 * @param size total size of the payload of the message (no header)
 * @param error reference for error-output
 * @param blockerId blocker-id in case that the message is a response
 *
 * @return 0, if failed, else the multiblock-id of the message
 */
uint64_t
MultiblockIO::sendOutgoingData(const void* data,
                               const uint64_t size,
                               ErrorContainer& error,
                               const uint64_t blockerId)
{
    // set or create id
    const uint64_t newMultiblockId = m_session->getRandId();

    // counter values
    uint64_t totalSize = size;
    uint64_t currentMessageSize = 0;
    uint32_t partCounter = 0;

    // static values
    const uint32_t totalPartNumber = static_cast<uint32_t>(totalSize / MAX_SINGLE_MESSAGE_SIZE) + 1;
    const uint8_t* dataPointer = static_cast<const uint8_t*>(data);

    while (totalSize != 0 && m_abort == false) {
        // get message-size base on the rest
        currentMessageSize = MAX_SINGLE_MESSAGE_SIZE;
        if (totalSize <= MAX_SINGLE_MESSAGE_SIZE) {
            currentMessageSize = totalSize;
        }
        totalSize -= currentMessageSize;

        // send single packet
        if (send_Data_Multi_Static(m_session,
                                   size,
                                   newMultiblockId,
                                   totalPartNumber,
                                   partCounter,
                                   dataPointer + (MAX_SINGLE_MESSAGE_SIZE * partCounter),
                                   static_cast<uint32_t>(currentMessageSize),
                                   error)
            == false)
        {
            return 0;
        }

        partCounter++;
    }

    if (m_abort) {
        return 0;
    }

    // finish multiblock-message
    if (send_Data_Multi_Finish(m_session, newMultiblockId, blockerId, error) == false) {
        return 0;
    }

    return newMultiblockId;
}

/**
 * @brief create new buffer for the message
 *
 * @param multiblockId id of the multiblock-message
 * @param size size for the new buffer
 *
 * @return false, if allocation failed, else true
 */
bool
MultiblockIO::createIncomingBuffer(const uint64_t multiblockId, const uint64_t size)
{
    // init new multiblock-message
    MultiblockBuffer newMultiblockMessage;
    newMultiblockMessage.incomingData = new Hanami::DataBuffer(calcBytesToBlocks(size));
    newMultiblockMessage.messageSize = size;
    newMultiblockMessage.multiblockId = multiblockId;

    // check if memory allocation was successful
    if (newMultiblockMessage.incomingData->data == nullptr) {
        delete newMultiblockMessage.incomingData;
        return false;
    }

    // put buffer into message-queue to be filled with incoming data
    m_lock.lock();
    m_incomingBuffer.insert(std::make_pair(multiblockId, newMultiblockMessage));
    m_lock.unlock();

    return true;
}

/**
 * @brief get incoming buffer by its id
 *
 * @param multiblockId id of the multiblock-message
 *
 * @return buffer, if found, else an empty-buffer-object
 */
MultiblockIO::MultiblockBuffer
MultiblockIO::getIncomingBuffer(const uint64_t multiblockId)
{
    std::lock_guard<std::mutex> guard(m_lock);

    std::map<uint64_t, MultiblockBuffer>::iterator it;
    it = m_incomingBuffer.find(multiblockId);
    if (it != m_incomingBuffer.end()) {
        return it->second;
    }

    MultiblockBuffer tempBuffer;
    return tempBuffer;
}

/**
 * @brief append data to the data-buffer for the multiblock-message
 *
 * @param multiblockId id of the multiblock-message
 * @param data pointer to the data
 * @param size number of bytes
 *
 * @return false, if session is not in the multiblock-transfer-state
 */
bool
MultiblockIO::writeIntoIncomingBuffer(const uint64_t multiblockId,
                                      const void* data,
                                      const uint64_t size)
{
    bool result = false;
    std::lock_guard<std::mutex> guard(m_lock);

    std::map<uint64_t, MultiblockBuffer>::iterator it;
    it = m_incomingBuffer.find(multiblockId);

    if (it != m_incomingBuffer.end()) {
        result = Hanami::addData_DataBuffer(*it->second.incomingData, data, size);
    }

    return result;
}

/**
 * @brief remove message form the inco-message-buffer, but without deleting the internal
 *        allocated memory.
 *
 * @param multiblockId it of the multiblock-message

 * @return true, if multiblock-id was found within the buffer, else false
 */
bool
MultiblockIO::removeMultiblockBuffer(const uint64_t multiblockId)
{
    std::lock_guard<std::mutex> guard(m_lock);

    std::map<uint64_t, MultiblockBuffer>::iterator it;
    it = m_incomingBuffer.find(multiblockId);
    if (it != m_incomingBuffer.end()) {
        m_incomingBuffer.erase(it);
        return true;
    }

    return false;
}

}  // namespace Hanami
