/**
 * @file        save_cluster_state.h
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

#ifndef SAVECLUSTERSTATE_H
#define SAVECLUSTERSTATE_H

#include <libKitsunemimiCommon/threading/event.h>
#include <libKitsunemimiJson/json_item.h>

class Cluster;

namespace Kitsunemimi {
namespace Hanami {
class HanamiMessagingClient;
}
}

class SaveCluster_State
        : public Kitsunemimi::Event
{
public:
    SaveCluster_State(Cluster* cluster);
    ~SaveCluster_State();

    bool processEvent();

private:
    Cluster* m_cluster = nullptr;

    bool sendData(const std::string &snapshotUuid,
                  const std::string &fileUuid,
                  Kitsunemimi::ErrorContainer &error);
};

#endif // SAVECLUSTERSTATE_H
