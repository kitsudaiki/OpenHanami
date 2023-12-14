/**
 * @file        save_cluster_state.cpp
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

#include "save_cluster_state.h"

#include <config.h>
#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/task.h>
#include <database/checkpoint_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>

extern "C" void copyFromGpu_CUDA(PointerHandler* gpuPointer,
                                 NeuronBlock* neuronBlocks,
                                 const uint32_t numberOfNeuronBlocks,
                                 SynapseBlock* synapseBlocks,
                                 const uint32_t numberOfSynapseBlocks);

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
SaveCluster_State::SaveCluster_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
SaveCluster_State::~SaveCluster_State() {}

/**
 * @brief prcess event
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::processEvent()
{
    bool result = false;
    Hanami::ErrorContainer error;

    do {
        Task* actualTask = m_cluster->getActualTask();
        const uint64_t totalSize = m_cluster->clusterData.usedBufferSize;

        // send checkpoint to shiori
        std::string fileUuid = "";
        // checkpoints are created by another internal process, which gives the id's not in the
        // context object, but as normal values
        UserContext userContext;
        userContext.userId = actualTask->userId;
        userContext.projectId = actualTask->projectId;

        // get directory to store data from config
        bool success = false;
        std::filesystem::path targetFilePath
            = GET_STRING_CONFIG("storage", "checkpoint_location", success);
        if (success == false) {
            error.addMessage("checkpoint-location to store checkpoint is missing in the config");
            break;
        }

        // build absolut file-path to store the file
        targetFilePath = targetFilePath
                         / std::filesystem::path(actualTask->uuid.toString() + "_checkpoint_"
                                                 + actualTask->userId);

        // register in database
        json dbEntry;
        dbEntry["uuid"] = actualTask->uuid.toString();
        dbEntry["name"] = actualTask->checkpointName;
        dbEntry["location"] = targetFilePath.generic_string();
        dbEntry["project_id"] = actualTask->projectId;
        dbEntry["owner_id"] = actualTask->userId;
        dbEntry["visibility"] = "private";

        // add to database
        if (CheckpointTable::getInstance()->addCheckpoint(dbEntry, userContext, error) == false) {
            break;
        }

        // write data of cluster to disc
        if (writeData(targetFilePath, totalSize, error) == false) {
            break;
        }

        result = true;
        break;
    }
    while (true);

    m_cluster->goToNextState(FINISH_TASK);

    if (result == false) {
        error.addMessage("Failed to create checkpoint of cluster with UUID '" + m_cluster->getUuid()
                         + "'");
        // TODO: cleanup in error-case
        // TODO: give the user a feedback by setting the task to failed-state
    }

    return result;
}

/**
 * @brief send all data of the checkpoint to shiori
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeData(const std::string& filePath,
                             const uint64_t fileSize,
                             Hanami::ErrorContainer& error)
{
    /*if (HanamiRoot::useCuda) {
        copyFromGpu_CUDA(&m_cluster->gpuPointer,
                         m_cluster->neuronBlocks,
                         m_cluster->clusterHeader->neuronBlocks.count,
                         m_cluster->synapseBlocks,
                         m_cluster->clusterHeader->synapseBlocks.count);
    }*/

    const uint64_t clusterSize = m_cluster->getDataSize();
    uint64_t position = 0;

    Hanami::BinaryFile checkpointFile(filePath);
    if (checkpointFile.allocateStorage(clusterSize, error) == false) {
        error.addMessage("Failed to allocate '" + std::to_string(fileSize)
                         + "' bytes for checkpointfile at path '" + filePath + "'");
        return false;
    }

    // create header
    CheckpointHeader header;
    header.setName(m_cluster->getName());
    header.setUuid(m_cluster->clusterHeader->uuid);
    header.metaSize = m_cluster->clusterData.totalBufferSize;
    header.blockSize = m_cluster->getDataSize() - header.metaSize;

    // write header of cluster to file
    if (checkpointFile.writeDataIntoFile(&header, position, sizeof(CheckpointHeader), error)
        == false)
    {
        error.addMessage("Failed to write cluster-header for checkpoint into file '" + filePath
                         + "'");
        return false;
    }
    position += sizeof(CheckpointHeader);

    // write static data of cluster to file
    if (checkpointFile.writeDataIntoFile(
            m_cluster->clusterData.data, position, m_cluster->clusterData.totalBufferSize, error)
        == false)
    {
        error.addMessage("Failed to write cluster-meta for checkpoint into file '" + filePath
                         + "'");
        return false;
    }
    position += m_cluster->clusterData.totalBufferSize;

    // write bricks to file
    for (uint64_t i = 0; i < m_cluster->clusterHeader->bricks.count; i++) {
        const uint64_t numberOfConnections = m_cluster->bricks[i].connectionBlocks.size();
        for (uint64_t c = 0; c < numberOfConnections; c++) {
            // write connection-blocks of brick to file
            ConnectionBlock* connectionBlock = &m_cluster->bricks[i].connectionBlocks[c];
            if (checkpointFile.writeDataIntoFile(
                    connectionBlock, position, sizeof(ConnectionBlock), error)
                == false)
            {
                error.addMessage("Failed to write connection-blocks for checkpoint into file '"
                                 + filePath + "'");
                return false;
            }
            position += sizeof(ConnectionBlock);

            // write synapse-blocks of brick to file
            SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::m_synapseBlocks);
            if (checkpointFile.writeDataIntoFile(
                    &synapseBlocks[connectionBlock->targetSynapseBlockPos],
                    position,
                    sizeof(SynapseBlock),
                    error)
                == false)
            {
                error.addMessage("Failed to write synapse-blocks for checkpoint into file '"
                                 + filePath + "'");
                return false;
            }
            position += sizeof(SynapseBlock);
        }
    }

    return true;
}
