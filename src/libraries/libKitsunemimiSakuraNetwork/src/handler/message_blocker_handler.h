/**
 * @file       message_blocker_handler.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_BLOCKER_HANDLER_H
#define KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_BLOCKER_HANDLER_H

#include <vector>
#include <iostream>

#include <libKitsunemimiCommon/threading/thread.h>

namespace Kitsunemimi::Sakura
{
class Session;

class MessageBlockerHandler
        : public Kitsunemimi::Thread
{
public:
    MessageBlockerHandler();
    ~MessageBlockerHandler();

    DataBuffer* blockMessage(const uint64_t blockerId,
                             const uint64_t blockerTimeout,
                             Session* session);
    bool releaseMessage(const uint64_t blockerId,
                        DataBuffer* data);

protected:
    void run();

private:
    struct MessageBlocker
    {
        Session* session = nullptr;
        uint64_t blockerId = 0;
        uint64_t timer = 0;
        std::mutex cvMutex;
        std::condition_variable cv;
        DataBuffer* responseData = nullptr;
    };

    std::vector<MessageBlocker*> m_messageList;

    bool releaseMessageInList(const uint64_t blockerId,
                              DataBuffer* data);
    DataBuffer* removeMessageFromList(const uint64_t blockerId);
    void clearList();
    void makeTimerStep();
};

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_BLOCKER_HANDLER_H
