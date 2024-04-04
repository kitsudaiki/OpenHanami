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

#include <hanami_common/logger.h>
#include <hanami_common/threading/event.h>

#include <nlohmann/json.hpp>

class Cluster;
class Task;
struct CheckpointHeader;

class RestoreCluster_State : public Hanami::Event
{
   public:
    RestoreCluster_State(Cluster* cluster);
    ~RestoreCluster_State();

    bool processEvent();

   private:
    Cluster* m_cluster = nullptr;

    bool restoreClusterFromCheckpoint(Task* currentTask, Hanami::ErrorContainer& error);

    bool restoreClusterFromFile(const std::string fileLocation, Hanami::ErrorContainer& error);
    bool restoreClusterHeader(const CheckpointHeader& header,
                              uint8_t* u8Data,
                              Hanami::ErrorContainer& error);

    bool restoreBricks(const CheckpointHeader& header,
                       uint8_t* u8Data,
                       Hanami::ErrorContainer& error);

    bool restoreNeuronBlocks(const CheckpointHeader& header,
                             uint8_t* u8Data,
                             Hanami::ErrorContainer& error);

    bool restoreConnectionBlocks(const CheckpointHeader& header,
                                 uint8_t* u8Data,
                                 Hanami::ErrorContainer& error);
};

#endif  // HANAMI_RESTORECLUSTERSTATE_H
