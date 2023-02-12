/**
 * @file        messaging_event_queue.cpp
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

#include "messaging_event_queue.h"

#include <libKitsunemimiCommon/logger.h>

#include <message_handling/messaging_event.h>

namespace Kitsunemimi::Hanami
{

Kitsunemimi::Hanami::MessagingEventQueue* MessagingEventQueue::m_instance = nullptr;

/**
 * @brief constructor
 */
MessagingEventQueue::MessagingEventQueue(const std::string &threadName)
    : Kitsunemimi::Thread(threadName)
{}

/**
 * @brief get instance of event-queue
 *
 * @return pointer to the instance of the event-queu
 */
MessagingEventQueue*
MessagingEventQueue::getInstance()
{
    if(m_instance == nullptr)
    {
        m_instance = new MessagingEventQueue("MessagingEventQueue");
        m_instance->startThread();
    }

    return m_instance;
}

/**
 * @brief run event-processing thread
 */
void
MessagingEventQueue::run()
{
    while(m_abort == false)
    {
        // get event
        Event* event = getEventFromQueue();
        if(event == nullptr)
        {
            // sleep if no event exist in the queue
            sleepThread(10000);
        }
        else
        {
            LOG_DEBUG("process messaging event");
            event->processEvent();
            delete event;
        }
    }
}

}
