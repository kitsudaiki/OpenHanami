/**
 * @file        messaging_event.h
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

#ifndef MESSAGING_EVENT_H
#define MESSAGING_EVENT_H

#include <libKitsunemimiCommon/threading/event.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiHanamiCommon/structs.h>

namespace Kitsunemimi
{
class JsonItem;

namespace Sakura {
class Session;
}
namespace Hanami
{
struct BlossomStatus;

class MessagingEvent
        : public Event
{
public:
    MessagingEvent(const HttpRequestType httpType,
                   const std::string &treeId,
                   const std::string &inputValues,
                   Kitsunemimi::Sakura::Session* session,
                   const uint64_t blockerId);
    ~MessagingEvent();

protected:
    bool processEvent();

private:
    uint64_t m_blockerId = 0;
    Kitsunemimi::Sakura::Session* m_session = nullptr;
    std::string m_targetId = "";
    std::string m_inputValues = "";
    HttpRequestType m_httpType = GET_TYPE;

    void sendResponseMessage(const bool success,
                             const HttpResponseTypes responseType,
                             const std::string &message,
                             Kitsunemimi::Sakura::Session* session,
                             const uint64_t blockerId,
                             ErrorContainer &error);
    bool trigger(DataMap &resultingItems,
                 JsonItem &inputValues,
                 Kitsunemimi::Hanami::BlossomStatus &status,
                 const EndpointEntry &endpoint,
                 ErrorContainer &error);

    void sendErrorMessage(const DataMap &context,
                          const JsonItem &inputValues,
                          const std::string &errorMessage);
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // MESSAGING_EVENT_H
