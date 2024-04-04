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
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/cuda/cuda_host.h>
#include <core/processing/logical_host.h>
#include <database/checkpoint_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>

extern "C" void copyFromGpu_CUDA(CudaPointerHandle* gpuPointer,
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
    Hanami::ErrorContainer error;
    Task* currentTask = m_cluster->getCurrentTask();

    const bool success = saveClusterToCheckpoint(currentTask, error);

    m_cluster->goToNextState(FINISH_TASK);

    if (success == false) {
        error.addMessage("Failed to create checkpoint of cluster with UUID '" + m_cluster->getUuid()
                         + "'");
        // TODO: cleanup in error-case
        // TODO: give the user a feedback by setting the task to failed-state
    }

    return success;
}

/**
 * @brief SaveCluster_State::saveClusterToCheckpoint
 * @param currentTask
 * @param error
 * @return
 */
bool
SaveCluster_State::saveClusterToCheckpoint(Task* currentTask, Hanami::ErrorContainer& error)
{
    // send checkpoint to shiori
    std::string fileUuid = "";
    // checkpoints are created by another internal process, which gives the id's not in the
    // context object, but as normal values
    UserContext userContext;
    userContext.userId = currentTask->userId;
    userContext.projectId = currentTask->projectId;

    // get directory to store data from config
    bool success = false;
    std::filesystem::path targetFilePath
        = GET_STRING_CONFIG("storage", "checkpoint_location", success);
    if (success == false) {
        error.addMessage("checkpoint-location to store checkpoint is missing in the config");
        return false;
    }

    // build absolut file-path to store the file
    targetFilePath = targetFilePath
                     / std::filesystem::path(currentTask->uuid.toString() + "_checkpoint_"
                                             + currentTask->userId);

    // register in database
    json dbEntry;
    dbEntry["uuid"] = currentTask->uuid.toString();
    dbEntry["name"] = currentTask->checkpointName;
    dbEntry["location"] = targetFilePath.generic_string();
    dbEntry["project_id"] = currentTask->projectId;
    dbEntry["owner_id"] = currentTask->userId;
    dbEntry["visibility"] = "private";

    // add to database
    if (CheckpointTable::getInstance()->addCheckpoint(dbEntry, userContext, error) == false) {
        return false;
    }

    // write data of cluster to disc
    m_cluster->attachedHost->syncWithHost(m_cluster);
    if (writeCheckpointToFile(targetFilePath, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief write the checkpoint of the cluster into a local file
 *
 * @param filePath path to the file, where the checkpoint should be written into
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeCheckpointToFile(const std::string& filePath, Hanami::ErrorContainer& error)
{
    const uint64_t totalFileSize = m_cluster->getDataSize() + sizeof(CheckpointHeader);

    // initialize checkpoint-file
    Hanami::BinaryFile checkpointFile(filePath);
    if (checkpointFile.allocateStorage(totalFileSize, error) == false) {
        error.addMessage("Failed to allocate '" + std::to_string(totalFileSize)
                         + "' bytes for checkpointfile at path '" + filePath + "'");
        return false;
    }

    // create header
    CheckpointHeader header;
    header.setName(m_cluster->getName());
    header.setUuid(m_cluster->clusterHeader.uuid);
    header.fileSize = totalFileSize;
    header.numberOfBricks = m_cluster->bricks.size();
    header.numberOfNeuronBrocks = m_cluster->neuronBlocks.size();

    uint64_t position = sizeof(CheckpointHeader);

    // cluster-header
    header.clusterHeaderPos = position;
    if (writeClusterHeaderToFile(checkpointFile, position, error) == false) {
        return false;
    }

    // neuron-blocks
    header.neuronBlocksPos = position;
    if (writeNeuronBlocksToFile(checkpointFile, position, error) == false) {
        return false;
    }

    // bricks
    header.bricksPos = position;
    if (writeBricksToFile(checkpointFile, position, error) == false) {
        return false;
    }

    // connection-blocks
    header.connectionBlocks = position;
    if (writeConnectionBlocksOfBricksToFile(checkpointFile, position, error) == false) {
        return false;
    }

    // write header of cluster to file
    if (checkpointFile.writeDataIntoFile(&header, 0, sizeof(CheckpointHeader), error) == false) {
        error.addMessage("Failed to write cluster-header for checkpoint into file");
        return false;
    }

    return true;
}

/**
 * @brief write cluster-header into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeClusterHeaderToFile(Hanami::BinaryFile& file,
                                            uint64_t& position,
                                            Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = sizeof(ClusterHeader);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&m_cluster->clusterHeader, position, numberOfBytes, error) == false)
    {
        error.addMessage("Failed to write cluster-header for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write bricks into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeBricksToFile(Hanami::BinaryFile& file,
                                     uint64_t& position,
                                     Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = m_cluster->bricks.size() * sizeof(Brick);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&m_cluster->bricks[0], position, numberOfBytes, error) == false) {
        error.addMessage("Failed to write bricks for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write neuron-blocks into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeNeuronBlocksToFile(Hanami::BinaryFile& file,
                                           uint64_t& position,
                                           Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = m_cluster->neuronBlocks.size() * sizeof(NeuronBlock);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&m_cluster->neuronBlocks[0], position, numberOfBytes, error)
        == false)
    {
        error.addMessage("Failed to write neuron-blocks for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write connection-blocks of the bricks into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeConnectionBlocksOfBricksToFile(Hanami::BinaryFile& file,
                                                       uint64_t& position,
                                                       Hanami::ErrorContainer& error)
{
    for (uint64_t i = 0; i < m_cluster->bricks.size(); i++) {
        const uint64_t numberOfConnections = m_cluster->bricks[i].connectionBlocks->size();
        for (uint64_t c = 0; c < numberOfConnections; c++) {
            if (writeConnectionBlockToFile(file, position, i, c, error) == false) {
                return true;
            }
        }
    }

    return true;
}

/**
 * @brief write a block of a brick into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param brickId id of the brick
 * @param blockid id of the block within the brick
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeConnectionBlockToFile(Hanami::BinaryFile& file,
                                              uint64_t& position,
                                              const uint64_t brickId,
                                              const uint64_t blockid,
                                              Hanami::ErrorContainer& error)
{
    // write connection-blocks of brick to file
    ConnectionBlock* connectionBlock = &m_cluster->bricks[brickId].connectionBlocks->at(blockid);
    if (file.writeDataIntoFile(connectionBlock, position, sizeof(ConnectionBlock), error) == false)
    {
        error.addMessage("Failed to write connection-blocks for checkpoint into file");
        return false;
    }
    position += sizeof(ConnectionBlock);

    // write synapse-blocks of brick to file
    if (writeSynapseBlockToFile(file, position, connectionBlock->targetSynapseBlockPos, error)
        == false)
    {
        return false;
    }

    return true;
}

/**
 * @brief write synapse-block into the file
 *
 * @param file reference to file-handler
 * @param position reference for position-counter to identify the position where to write into the
 *                 file
 * @param targetSynapseBlockPos position of the synapse-block within the global buffer
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeSynapseBlockToFile(Hanami::BinaryFile& file,
                                           uint64_t& position,
                                           const uint64_t targetSynapseBlockPos,
                                           Hanami::ErrorContainer& error)
{
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(m_cluster->attachedHost->synapseBlocks);
    if (file.writeDataIntoFile(
            &synapseBlocks[targetSynapseBlockPos], position, sizeof(SynapseBlock), error)
        == false)
    {
        error.addMessage("Failed to write synapse-blocks for checkpoint into file");
        return false;
    }
    position += sizeof(SynapseBlock);

    return true;
}
