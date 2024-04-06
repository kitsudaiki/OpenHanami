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

/**
 * @brief constructor
 *
 * @param localId identifier starting with 0 within the physical host and with the type of host
 */
LogicalHost::LogicalHost(const uint32_t localId) : Hanami::Thread("ProcessingUnit")
{
    m_localId = localId;
    m_uuid = generateUuid().toString();
}

/**
 * @brief destructor
 */
LogicalHost::~LogicalHost() {}

/**
 * @brief get host-type of this logical host (cpu, cuda, ...)
 */
LogicalHost::HostType
LogicalHost::getHostType() const
{
    return m_hostType;
}

/**
 * @brief get uuid of the host for identification
 */
const std::string
LogicalHost::getUuid() const
{
    return m_uuid;
}

/**
 * @brief get amount of initialy defined available memory
 */
uint64_t
LogicalHost::getTotalMemory()
{
    return m_totalMemory;
}

/**
 * @brief run loop to process all scheduled cluster
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

/**
 * @brief get position of the highest output-position
 *
 * @param cluster output-cluster to check
 *
 * @return position of the highest output.
 */
uint32_t
getHighestOutput(const Cluster& cluster)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for (uint32_t outputNeuronId = 0; outputNeuronId < cluster.clusterHeader.numberOfOutputs;
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
 * @brief in case of a request-task the output is either written into a request-result in case of a
 *        task or is send via websocket to a client in case of direct-io
 *
 * @param cluster cluster to handle
 */
void
handleClientOutput(const Cluster& cluster)
{
    // send output back if a client-connection is set
    if (cluster.msgClient != nullptr) {
        sendClusterOutputMessage(&cluster);
    }
    else {
        Task* actualTask = cluster.getCurrentTask();
        const uint64_t cycle = actualTask->actualCycle;
        if (actualTask->type == IMAGE_REQUEST_TASK) {
            // TODO: check for cluster-state instead of client
            const uint32_t hightest = getHighestOutput(cluster);
            actualTask->resultData[cycle] = static_cast<long>(hightest);
        }
        else if (actualTask->type == TABLE_REQUEST_TASK) {
            float val = 0.0f;
            for (uint64_t i = 0; i < cluster.clusterHeader.numberOfOutputs; i++) {
                const float temp = actualTask->resultData[cycle];
                val = temp + cluster.outputValues[i];
                actualTask->resultData[cycle] = val;
            }
        }
    }
}
