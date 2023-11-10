/**
 * @file       message_blocker_handler.cpp
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

#include "message_blocker_handler.h"

#include <hanami_network/session.h>

namespace Hanami
{

/**
 * @brief constructor
 */
MessageBlockerHandler::MessageBlockerHandler() : Hanami::Thread("MessageBlockerHandler") {}

/**
 * @brief destructor
 */
MessageBlockerHandler::~MessageBlockerHandler() { clearList(); }

/**
 * @brief MessageBlockerHandler::blockMessage
 *
 * @param blockerId id ot identify the entry within the blocker-handler
 * @param blockerTimeout time until a timeout appear for the message in seconds
 * @param session pointer to the session for error-callback in case of a timeout
 * @return
 */
DataBuffer*
MessageBlockerHandler::blockMessage(const uint64_t blockerId,
                                    const uint64_t blockerTimeout,
                                    Session* session)
{
    // init new blocker entry
    MessageBlocker* messageBlocker = new MessageBlocker();
    messageBlocker->blockerId = blockerId;
    messageBlocker->timer = blockerTimeout;
    messageBlocker->session = session;

    // add to waiting-list
    spinLock();
    m_messageList.push_back(messageBlocker);
    spinUnlock();

    // release thread
    std::unique_lock<std::mutex> lock(messageBlocker->cvMutex);
    messageBlocker->cv.wait(lock);

    // remove from list and return result
    spinLock();
    DataBuffer* result = removeMessageFromList(blockerId);
    spinUnlock();

    return result;
}

/**
 * @brief MessageBlockerHandler::releaseMessage
 * @param blockerId
 * @param data data-buffer, which comes from the other side and should be returned by the
 *             blocked thread, which called the request-method within the session
 *
 * @return true, if blocker-id was found in the list of blocked threads
 */
bool
MessageBlockerHandler::releaseMessage(const uint64_t blockerId, DataBuffer* data)
{
    bool result = false;

    spinLock();
    result = releaseMessageInList(blockerId, data);
    spinUnlock();

    return result;
}

/**
 * @brief AnswerHandler::run
 */
void
MessageBlockerHandler::run()
{
    while (m_abort == false) {
        makeTimerStep();

        // sleep for 1 second
        sleepThread(1000000);
    }
}

/**
 * @brief MessageBlockerHandler::releaseMessageInList
 *
 * @param blockerId
 * @param data
 *
 * @return
 */
bool
MessageBlockerHandler::releaseMessageInList(const uint64_t blockerId, DataBuffer* data)
{
    for (MessageBlocker* blocker : m_messageList) {
        if (blocker->blockerId == blockerId) {
            blocker->responseData = data;
            blocker->cv.notify_one();
            return true;
        }
    }

    return false;
}

/**
 * @brief AnswerHandler::removeMessageFromList
 * @param blockerId
 * @return
 */
DataBuffer*
MessageBlockerHandler::removeMessageFromList(const uint64_t blockerId)
{
    std::vector<MessageBlocker*>::iterator it;
    for (it = m_messageList.begin(); it != m_messageList.end(); it++) {
        MessageBlocker* tempItem = *it;
        if (tempItem->blockerId == blockerId) {
            if (m_messageList.size() > 1) {
                // swap with last and remove the last instead of erase the element direct
                // because this was is faster
                std::iter_swap(it, m_messageList.end() - 1);
                m_messageList.pop_back();
            }
            else {
                m_messageList.clear();
            }

            // get data-buffer from the blocker-entry and delete the entry afterwards
            DataBuffer* result = tempItem->responseData;
            tempItem->responseData = nullptr;
            delete tempItem;

            return result;
        }
    }

    return nullptr;
}

/**
 * @brief AnswerHandler::clearList
 */
void
MessageBlockerHandler::clearList()
{
    spinLock();

    // release all threads
    for (MessageBlocker* blocker : m_messageList) {
        blocker->cv.notify_one();
        delete blocker;
    }

    // clear list
    m_messageList.clear();

    spinUnlock();
}

/**
 * @brief MessageBlockerHandler::makeTimerStep
 */
void
MessageBlockerHandler::makeTimerStep()
{
    spinLock();
    for (MessageBlocker* blocker : m_messageList) {
        blocker->timer -= 1;

        if (blocker->timer == 0) {
            removeMessageFromList(blocker->blockerId);
            releaseMessageInList(blocker->blockerId, nullptr);

            const std::string err = "TIMEOUT of request: " + std::to_string(blocker->blockerId);
            blocker->session->m_processError(
                blocker->session, Session::errorCodes::MESSAGE_TIMEOUT, err);
        }
    }

    spinUnlock();
}

}  // namespace Hanami
