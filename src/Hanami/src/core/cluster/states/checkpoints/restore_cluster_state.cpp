/**
 * @file        restore_cluster_state.cpp
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

#include "restore_cluster_state.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/task.h>
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/logical_host.h>
#include <core/processing/physical_host.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
RestoreCluster_State::RestoreCluster_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
RestoreCluster_State::~RestoreCluster_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
RestoreCluster_State::processEvent()
{
    Task* currentTask = m_cluster->getCurrentTask();
    Hanami::ErrorContainer error;

    const bool success = restoreClusterFromCheckpoint(currentTask, error);

    m_cluster->goToNextState(FINISH_TASK);

    return success;
}

/**
 * @brief RestoreCluster_State::restoreCluster
 * @param currentTask
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreClusterFromCheckpoint(Task* currentTask, Hanami::ErrorContainer& error)
{
    // get meta-infos of dataset from shiori
    json parsedCheckpointInfo;
    try {
        parsedCheckpointInfo = json::parse(currentTask->checkpointInfo);
    }
    catch (const json::parse_error& ex) {
        error.addMessage("json-parser error: " + std::string(ex.what()));
        return false;
    }

    // get other information
    if (parsedCheckpointInfo.contains("location") == false) {
        return false;
    }
    const std::string location = parsedCheckpointInfo["location"];

    if (restoreClusterFromFile(location, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief RestoreCluster_State::restoreClusterFromFile
 * @param fileLocation
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreClusterFromFile(const std::string fileLocation,
                                             Hanami::ErrorContainer& error)
{
    const std::string originalUuid = m_cluster->getUuid();

    // get checkpoint-data
    Hanami::BinaryFile checkpointFile(fileLocation);
    Hanami::DataBuffer checkpointBuffer;
    if (checkpointFile.readCompleteFile(checkpointBuffer, error) == false) {
        error.addMessage("failed to load checkpoint-data");
        return false;
    }

    uint8_t* u8Data = static_cast<uint8_t*>(checkpointBuffer.data);

    // convert checkpoint-header
    CheckpointHeader header;
    memcpy(&header, &u8Data[0], sizeof(CheckpointHeader));

    if (restoreClusterHeader(header, u8Data, error) == false) {
        return false;
    }

    if (restoreNeuronBlocks(header, u8Data, error) == false) {
        return false;
    }

    if (restoreBricks(header, u8Data, error) == false) {
        return false;
    }

    if (restoreConnectionBlocks(header, u8Data, error) == false) {
        return false;
    }

    // update uuid
    strncpy(m_cluster->clusterHeader.uuid.uuid, originalUuid.c_str(), originalUuid.size());

    return true;
}

/**
 * @brief RestoreCluster_State::restoreClusterHeader
 * @param header
 * @param u8Data
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreClusterHeader(const CheckpointHeader& header,
                                           uint8_t* u8Data,
                                           Hanami::ErrorContainer& error)
{
    const uint64_t position = header.clusterHeaderPos;
    const uint64_t size = sizeof(ClusterHeader);

    memcpy(&m_cluster->clusterHeader, &u8Data[position], size);

    return true;
}

/**
 * @brief RestoreCluster_State::restoreNeuronBlocks
 * @param header
 * @param u8Data
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreNeuronBlocks(const CheckpointHeader& header,
                                          uint8_t* u8Data,
                                          Hanami::ErrorContainer& error)
{
    const uint64_t position = header.neuronBlocksPos;
    const uint64_t size = header.numberOfNeuronBrocks * sizeof(NeuronBlock);

    m_cluster->neuronBlocks.clear();
    m_cluster->neuronBlocks.resize(header.numberOfNeuronBrocks);

    memcpy(&m_cluster->neuronBlocks[0], &u8Data[position], size);

    return true;
}

/**
 * @brief RestoreCluster_State::restoreBricks
 * @param header
 * @param u8Data
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreBricks(const CheckpointHeader& header,
                                    uint8_t* u8Data,
                                    Hanami::ErrorContainer& error)
{
    const uint64_t position = header.bricksPos;
    const uint64_t size = header.numberOfBricks * sizeof(Brick);

    m_cluster->bricks.clear();
    m_cluster->bricks.resize(header.numberOfBricks);

    memcpy(&m_cluster->bricks[0], &u8Data[position], size);

    for (Brick& brick : m_cluster->bricks) {
        brick.connectionBlocks = new std::vector<ConnectionBlock>();
    }

    return true;
}

/**
 * @brief RestoreCluster_State::restoreConnectionBlocks
 * @param header
 * @param u8Data
 * @param error
 * @return
 */
bool
RestoreCluster_State::restoreConnectionBlocks(const CheckpointHeader& header,
                                              uint8_t* u8Data,
                                              Hanami::ErrorContainer& error)
{
    uint64_t position = header.connectionBlocks;

    // get initial logical host
    LogicalHost* host = HanamiRoot::physicalHost->getFirstHost();
    if (host == nullptr) {
        error.addMessage("No logical host found for new cluster.");
        return false;
    }

    // convert payload of bricks
    for (Brick& brick : m_cluster->bricks) {
        brick.connectionBlocks->resize(brick.dimX * brick.dimY);

        const uint64_t numberOfConnections = brick.connectionBlocks->size();
        for (uint64_t c = 0; c < numberOfConnections; c++) {
            // convert connection-block
            memcpy(&brick.connectionBlocks[0][c], &u8Data[position], sizeof(ConnectionBlock));
            position += sizeof(ConnectionBlock);

            // convert synapse-block
            SynapseBlock newSynapseBlock;
            memcpy(&newSynapseBlock, &u8Data[position], sizeof(SynapseBlock));
            const uint64_t itemPos = host->synapseBlocks.addNewItem(newSynapseBlock);
            if (itemPos == UNINIT_STATE_64) {
                error.addMessage("failed allocate synapse-block for checkpoint");
                return false;
            }

            // write new position into the related connection-block
            brick.connectionBlocks[0][c].targetSynapseBlockPos = itemPos;

            position += sizeof(SynapseBlock);
        }
    }

    return true;
}
