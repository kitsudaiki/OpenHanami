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

#ifndef HANAMI_SAVECLUSTERSTATE_H
#define HANAMI_SAVECLUSTERSTATE_H

#include <hanami_common/logger.h>
#include <hanami_common/threading/event.h>

class Cluster;
class Task;

namespace Hanami
{
class BinaryFile;
}

class SaveCluster_State : public Hanami::Event
{
   public:
    SaveCluster_State(Cluster* cluster);
    ~SaveCluster_State();

    bool processEvent();

   private:
    Cluster* m_cluster = nullptr;

    bool writeCheckpointToFile(const std::string& filePath, Hanami::ErrorContainer& error);

    bool writeClusterHeaderToFile(Hanami::BinaryFile& file,
                                  uint64_t& position,
                                  Hanami::ErrorContainer& error);

    bool writeBricksToFile(Hanami::BinaryFile& file,
                           uint64_t& position,
                           Hanami::ErrorContainer& error);

    bool writeNeuronBlocksToFile(Hanami::BinaryFile& file,
                                 uint64_t& position,
                                 Hanami::ErrorContainer& error);

    bool writeConnectionBlocksOfBricksToFile(Hanami::BinaryFile& file,
                                             uint64_t& position,
                                             Hanami::ErrorContainer& error);

    bool writeConnectionBlockToFile(Hanami::BinaryFile& file,
                                    uint64_t& position,
                                    const uint64_t brickId,
                                    const uint64_t blockid,
                                    Hanami::ErrorContainer& error);

    bool writeSynapseBlockToFile(Hanami::BinaryFile& file,
                                 uint64_t& position,
                                 const uint64_t targetSynapseBlockPos,
                                 Hanami::ErrorContainer& error);
    bool saveClusterToCheckpoint(Task* currentTask, Hanami::ErrorContainer& error);
};

#endif  // HANAMI_SAVECLUSTERSTATE_H
