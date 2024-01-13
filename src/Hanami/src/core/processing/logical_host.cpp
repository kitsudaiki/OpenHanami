/**
 * @file        logical_host.cpp
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

#include "logical_host.h"

#include <api/websocket/cluster_io.h>
#include <core/cluster/cluster.h>
#include <core/processing/objects.h>
#include <hanami_common/buffer/item_buffer.h>

LogicalHost::LogicalHost(const uint32_t localId) : Hanami::Thread("ProcessingUnit")
{
    m_localId = localId;
    m_uuid = generateUuid().toString();
}

LogicalHost::~LogicalHost() {}

/**
 * @brief add cluster to queue
 *
 * @param newSegment cluster to add to queue
 */
void
LogicalHost::addClusterToQueue(Cluster* cluster)
{
    while (m_queue_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }
    m_clusterQueue.push_back(cluster);
    m_queue_lock.clear(std::memory_order_release);
}

/**
 * @brief get next cluster in the queue
 *
 * @return nullptr, if queue is empty, else next cluster in queue
 */
Cluster*
LogicalHost::getClusterFromQueue()
{
    Cluster* result = nullptr;

    while (m_queue_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }

    if (m_clusterQueue.size() > 0) {
        result = m_clusterQueue.front();
        m_clusterQueue.pop_front();
    }

    m_queue_lock.clear(std::memory_order_release);

    return result;
}

/**
 * @brief LogicalHost::getHostType
 * @return
 */
LogicalHost::HostType
LogicalHost::getHostType() const
{
    return m_hostType;
}

const std::string
LogicalHost::getUuid() const
{
    return m_uuid;
}

/**
 * @brief get position of the highest output-position
 *
 * @param cluster output-cluster to check
 *
 * @return position of the highest output.
 */
uint32_t
LogicalHost::getHighestOutput(const Cluster& cluster)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for (uint32_t outputNeuronId = 0; outputNeuronId < cluster.clusterHeader->numberOfOutputs;
         outputNeuronId++)
    {
        value = cluster.outputValues[outputNeuronId];
        if (value > hightest) {
            hightest = value;
            hightestPos = outputNeuronId;
        }
    }

    return hightestPos;
}

/**
 * @brief LogicalHost::handleClientOutput
 * @param cluster
 */
void
LogicalHost::handleClientOutput(const Cluster& cluster)
{
    // send output back if a client-connection is set
    if (cluster.msgClient != nullptr) {
        sendClusterOutputMessage(&cluster);
    }
    else {
        Task* actualTask = cluster.getActualTask();
        const uint64_t cycle = actualTask->actualCycle;
        if (actualTask->type == IMAGE_REQUEST_TASK) {
            // TODO: check for cluster-state instead of client
            const uint32_t hightest = getHighestOutput(cluster);
            actualTask->resultData[cycle] = static_cast<long>(hightest);
        }
        else if (actualTask->type == TABLE_REQUEST_TASK) {
            float val = 0.0f;
            for (uint64_t i = 0; i < cluster.clusterHeader->numberOfOutputs; i++) {
                const float temp = actualTask->resultData[cycle];
                val = temp + cluster.outputValues[i];
                actualTask->resultData[cycle] = val;
            }
        }
    }
}

/**
 * @brief run loop to process all available segments
 */
void
LogicalHost::run()
{
    Cluster* cluster = nullptr;

    while (m_abort == false) {
        cluster = getClusterFromQueue();
        if (cluster != nullptr) {
            // handle type of processing
            if (cluster->mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
                trainClusterForward(cluster);
            }
            else if (cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
                trainClusterBackward(cluster);
            }
            else {
                requestCluster(cluster);
                handleClientOutput(*cluster);
            }
            cluster->updateClusterState();
        }
        else {
            // if no segments are available then sleep
            sleepThread(1000);
        }
    }
}
