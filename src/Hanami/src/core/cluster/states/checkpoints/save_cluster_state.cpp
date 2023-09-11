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
#include <database/checkpoint_table.h>
#include <hanami_root.h>
#include <core/cluster/task.h>
#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>

#include <hanami_crypto/hashes.h>
#include <hanami_common/files/binary_file.h>


extern "C"
void
copyFromGpu_CUDA(PointerHandler* gpuPointer,
                 NeuronBlock* neuronBlocks,
                 const uint32_t numberOfNeuronBlocks,
                 SynapseBlock* synapseBlocks,
                 const uint32_t numberOfSynapseBlocks);

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
SaveCluster_State::SaveCluster_State(Cluster* cluster)
{
    m_cluster = cluster;
}

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

    do
    {
        Task* actualTask = m_cluster->getActualTask();
        const uint64_t totalSize = m_cluster->clusterData.buffer.usedBufferSize;

        // send checkpoint to shiori
        std::string fileUuid = "";
        // checkpoints are created by another internal process, which gives the id's not in the context
        // object, but as normal values
        UserContext userContext;
        userContext.userId = actualTask->userId;
        userContext.projectId = actualTask->projectId;

        // get directory to store data from config
        bool success = false;
        std::string targetFilePath = GET_STRING_CONFIG("storage", "checkpoint_location", success);
        if(success == false)
        {
            error.addMeesage("checkpoint-location to store checkpoint is missing in the config");
            break;
        }

        // build absolut file-path to store the file
        if(targetFilePath.at(targetFilePath.size() - 1) != '/') {
            targetFilePath.append("/");
        }
        targetFilePath.append(actualTask->uuid.toString() + "_checkpoint_" + actualTask->userId);

        // register in database
        json dbEntry;
        dbEntry["uuid"] = actualTask->uuid.toString();
        dbEntry["name"] = actualTask->checkpointName;
        dbEntry["location"] = targetFilePath;
        dbEntry["project_id"] = actualTask->projectId;
        dbEntry["owner_id"] = actualTask->userId;
        dbEntry["visibility"] = "private";

        // add to database
        if(CheckpointTable::getInstance()->addCheckpoint(dbEntry,
                                                                   userContext,
                                                                   error) == false)
        {
            break;
        }

        // write data of cluster to disc
        if(writeData(targetFilePath, totalSize, error) == false) {
            break;
        }

        result = true;
        break;
    }
    while(true);

    m_cluster->goToNextState(FINISH_TASK);

    if(result == false)
    {
        error.addMeesage("Failed to create checkpoint of cluster with UUID '"
                         + m_cluster->getUuid()
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
SaveCluster_State::writeData(const std::string &filePath,
                             const uint64_t fileSize,
                             Hanami::ErrorContainer &error)
{
    if(HanamiRoot::useCuda)
    {
        copyFromGpu_CUDA(&m_cluster->gpuPointer,
                         m_cluster->neuronBlocks,
                         m_cluster->clusterHeader->neuronBlocks.count,
                         m_cluster->synapseBlocks,
                         m_cluster->clusterHeader->synapseBlocks.count);
    }

    Hanami::BinaryFile checkpointFile(filePath);
    if(checkpointFile.allocateStorage(fileSize, error) == false)
    {
        error.addMeesage("Failed to allocate '"
                         + std::to_string(fileSize)
                         + "' bytes for checkpointfile at path '"
                         + filePath
                         + "'");
        return false;
    }

    // global byte-counter to identifiy the position within the complete checkpoint
    Hanami::DataBuffer* buffer = &m_cluster->clusterData.buffer;
    if(checkpointFile.writeDataIntoFile(buffer->data, 0, buffer->usedBufferSize, error) == false)
    {
        error.addMeesage("Failed to write cluster for checkpoint into file '"
                         + filePath
                         + "'");
        return false;
    }

    return true;
}
