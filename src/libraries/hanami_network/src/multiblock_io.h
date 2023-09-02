/**
 * @file       multiblock_io.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_IO_H
#define KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_IO_H

#include <iostream>
#include <assert.h>
#include <atomic>
#include <utility>
#include <deque>
#include <map>
#include <string>

#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/logger.h>

namespace Hanami
{
class Session;

class MultiblockIO
{
public:
    // multiblock-message
    struct MultiblockBuffer
    {
        uint64_t blockerId = 0;
        uint64_t multiblockId = 0;
        uint64_t messageSize = 0;
        uint32_t numberOfPackages = 0;
        uint32_t courrentPackage = 0;

        Hanami::DataBuffer* incomingData = nullptr;
    };

    MultiblockIO(Session* session);
    ~MultiblockIO();

    // create
    uint64_t sendOutgoingData(const void* data,
                              const uint64_t size,
                              ErrorContainer &error,
                              const uint64_t blockerId = 0);
    bool createIncomingBuffer(const uint64_t multiblockId,
                              const uint64_t size);

    // process incoming
    MultiblockBuffer getIncomingBuffer(const uint64_t multiblockId);
    bool writeIntoIncomingBuffer(const uint64_t multiblockId,
                                 const void* data,
                                 const uint64_t size);
    bool removeMultiblockBuffer(const uint64_t multiblockId);

private:
    Session* m_session = nullptr;
    bool m_abort = false;

    std::mutex m_lock;
    std::map<uint64_t, MultiblockBuffer> m_incomingBuffer;
};

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_MULTIBLOCK_IO_H
