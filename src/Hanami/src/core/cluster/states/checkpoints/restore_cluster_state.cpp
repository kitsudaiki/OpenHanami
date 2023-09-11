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
#include <hanami_root.h>
#include <core/cluster/task.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/statemachine_init.h>

#include <hanami_common/files/binary_file.h>
#include <hanami_crypto/hashes.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
RestoreCluster_State::RestoreCluster_State(Cluster* cluster)
{
    m_cluster = cluster;
}

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
    Task* actualTask = m_cluster->getActualTask();
    Hanami::ErrorContainer error;
    const std::string originalUuid = m_cluster->getUuid();

    // get meta-infos of data-set from shiori
    json parsedCheckpointInfo = json::parse(actualTask->checkpointInfo, nullptr, false);
    if (parsedCheckpointInfo.is_discarded())
    {
        std::cerr << "parse error" << std::endl;
        return false;
    }

    // get other information
    const std::string location = parsedCheckpointInfo["location"];

    // get checkpoint-data
    Hanami::BinaryFile checkpointFile(location);
    Hanami::DataBuffer checkpointBuffer;
    if(checkpointFile.readCompleteFile(checkpointBuffer, error) == false)
    {
        error.addMeesage("failed to load checkpoint-data");
        m_cluster->goToNextState(FINISH_TASK);
        return false;
    }

    // copy data of cluster
    const uint8_t* u8Data = static_cast<const uint8_t*>(checkpointBuffer.data);
    m_cluster->clusterData.initBuffer(u8Data, checkpointBuffer.usedBufferSize);
    if(reinitPointer(m_cluster, checkpointBuffer.usedBufferSize) == false)
    {
        error.addMeesage("failed to re-init pointer in cluster");
        m_cluster->goToNextState(FINISH_TASK);
        return false;
    }

    strncpy(m_cluster->clusterHeader->uuid.uuid,
            originalUuid.c_str(),
            originalUuid.size());
    m_cluster->goToNextState(FINISH_TASK);

    return true;
}
