/**
 * @file        messaging_event_queue.h
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

#ifndef MESSAGING_EVENT_QUEUE_H
#define MESSAGING_EVENT_QUEUE_H

#include <libKitsunemimiCommon/threading/thread.h>

namespace Kitsunemimi
{
namespace Hanami
{

class MessagingEventQueue
        : public Kitsunemimi::Thread
{
public:  
    static MessagingEventQueue* getInstance();

protected:
    void run();

private:
    MessagingEventQueue(const std::string &threadName);

    static MessagingEventQueue* m_instance;
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // MESSAGING_EVENT_QUEUE_H
