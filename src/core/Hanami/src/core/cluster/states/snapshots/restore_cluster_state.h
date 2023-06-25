/**
 * @file        restore_cluster_state.h
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

#ifndef HANAMI_RESTORECLUSTERSTATE_H
#define HANAMI_RESTORECLUSTERSTATE_H

#include <libKitsunemimiCommon/threading/event.h>
#include <libKitsunemimiJson/json_item.h>

class Cluster;

namespace Kitsunemimi::Hanami {
class HanamiMessagingClient;
}

class RestoreCluster_State
        : public Kitsunemimi::Event
{
public:
    RestoreCluster_State(Cluster* cluster);
    ~RestoreCluster_State();

    bool processEvent();

private:
    Cluster* m_cluster = nullptr;
    Kitsunemimi::Hanami::HanamiMessagingClient* m_client = nullptr;
};

#endif // HANAMI_RESTORECLUSTERSTATE_H
