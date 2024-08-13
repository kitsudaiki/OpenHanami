/**
 * @file        cpu_host.cpp
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

#include "cpu_host.h"

#include <core/processing/cluster_resize.h>
#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/cpu/worker_thread.h>
#include <hanami_config/config_handler.h>
#include <hanami_cpu/memory.h>
#include <hanami_hardware/cpu_core.h>
#include <hanami_hardware/cpu_package.h>
#include <hanami_hardware/cpu_thread.h>
#include <hanami_hardware/host.h>

/**
 * @brief constructor
 *
 * @param localId identifier starting with 0 within the physical host and with the type of host
 */
CpuHost::CpuHost(const uint32_t localId) : LogicalHost(localId)
{
    m_hostType = CPU_HOST_TYPE;
    initBuffer(localId);
    initWorkerThreads();
}

/**
 * @brief destructor
 */
CpuHost::~CpuHost() {}

/**
 * @brief CpuHost::addClusterToHost
 * @param cluster
 */
void
CpuHost::addClusterToHost(Cluster* cluster)
{
    if (cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
        WorkerTask task;
        task.cluster = cluster;
        task.hexagonId = cluster->hexagons.size() - 1;
        task.blockId = UNINIT_STATE_16;
        addWorkerTaskToQueue(task);
    }
    else {
        WorkerTask task;
        task.cluster = cluster;
        task.hexagonId = 0;
        task.blockId = UNINIT_STATE_16;
        addWorkerTaskToQueue(task);
    }
}

/**
 * @brief not implemented in this case
 */
Cluster*
CpuHost::getClusterFromQueue()
{
    return nullptr;
}

/**
 * @brief initialize synpase-block-buffer based on the avaialble size of memory
 *
 * @param id local device-id
 */
void
CpuHost::initBuffer(const uint32_t id)
{
    m_totalMemory = getFreeMemory();
    bool success = false;
    float memoryUsage = GET_FLOAT_CONFIG("processing", "use_of_free_memory", success);
    // TODO: handle amound of min and max value by ranges inside of the config-lib
    if (memoryUsage < 0.1f) {
        memoryUsage = 0.1f;
    }
    if (memoryUsage > 0.9f) {
        memoryUsage = 0.9f;
    }
    const uint64_t usedMemory = static_cast<float>(m_totalMemory) * memoryUsage;
    synapseBlocks.initBuffer<SynapseBlock>(usedMemory / sizeof(SynapseBlock));
    synapseBlocks.deleteAll();

    LOG_INFO("Initialized number of syanpse-blocks on cpu-device with id '" + std::to_string(id)
             + "': " + std::to_string(synapseBlocks.metaData->itemCapacity));
}

/**
 * @brief init processing-thread
 */
bool
CpuHost::initWorkerThreads()
{
    Host* host = Host::getInstance();
    CpuPackage* package = host->cpuPackages.at(m_localId);
    uint32_t threadCounter = 0;
    for (uint32_t coreId = 1; coreId < package->cpuCores.size(); coreId++) {
        for (uint32_t threadId = 0; threadId < package->cpuCores.at(coreId)->cpuThreads.size();
             threadId++)
        {
            CpuThread* thread = package->cpuCores.at(coreId)->cpuThreads.at(threadId);
            WorkerThread* newUnit = new WorkerThread(this);
            m_workerThreads.push_back(newUnit);
            newUnit->startThread();
            newUnit->bindThreadToCore(thread->threadId);
            threadCounter++;
        }
    }

    LOG_INFO("Initialized " + std::to_string(threadCounter) + " worker-threads");

    return true;
}

/**
 * @brief re-activate all blocked threads
 */
void
CpuHost::continueAllThreads()
{
    for (WorkerThread* worker : m_workerThreads) {
        worker->continueThread();
    }
}

/**
 * @brief move the data of a cluster to this host
 *
 * @param cluster cluster to move
 *
 * @return true, if successful, else false
 */
bool
CpuHost::moveCluster(Cluster* cluster)
{
    LogicalHost* originHost = cluster->attachedHost;
    SynapseBlock* cpuSynapseBlocks = Hanami::getItemData<SynapseBlock>(synapseBlocks);
    SynapseBlock tempBlock;

    // copy synapse-blocks from the old host to this one here
    for (Hexagon& hexagon : cluster->hexagons) {
        for (ConnectionBlock& block : hexagon.connectionBlocks) {
            if (block.targetSynapseBlockPos != UNINIT_STATE_64) {
                tempBlock = cpuSynapseBlocks[block.targetSynapseBlockPos];
                originHost->synapseBlocks.deleteItem(block.targetSynapseBlockPos);
                const uint64_t newPos = synapseBlocks.addNewItem(tempBlock);
                // TODO: make roll-back possible in error-case
                if (newPos == UNINIT_STATE_64) {
                    return false;
                }
                block.targetSynapseBlockPos = newPos;
            }
        }
    }

    cluster->attachedHost = this;

    return true;
}

/**
 * @brief empty function in this case
 */
void
CpuHost::syncWithHost(Cluster*)
{
}

/**
 * @brief remove synpase-blocks of a cluster from the host-buffer
 *
 * @param cluster cluster to clear
 */
void
CpuHost::removeCluster(Cluster* cluster)
{
    for (Hexagon& hexagon : cluster->hexagons) {
        for (ConnectionBlock& block : hexagon.connectionBlocks) {
            if (block.targetSynapseBlockPos != UNINIT_STATE_64) {
                synapseBlocks.deleteItem(block.targetSynapseBlockPos);
            }
        }
    }
}

/**
 * @brief empty in this case, because this is done by the worker-threads
 */
void
CpuHost::trainClusterForward(Cluster*)
{
}

/**
 * @brief empty in this case, because this is done by the worker-threads
 */
void
CpuHost::trainClusterBackward(Cluster*)
{
}

/**
 * @brief empty in this case, because this is done by the worker-threads
 */
void
CpuHost::requestCluster(Cluster*)
{
}

/**
 * @brief add cluster to queue
 *
 * @param cluster cluster to add to queue
 */
void
CpuHost::addWorkerTaskToQueue(const WorkerTask task)
{
    const std::lock_guard<std::mutex> lock(m_queue_lock);

    m_workerTaskQueue.push_back(task);
    continueAllThreads();
}

/**
 * @brief get next cluster in the queue
 *
 * @return nullptr, if queue is empty, else next cluster in queue
 */
const CpuHost::WorkerTask
CpuHost::getWorkerTaskFromQueue()
{
    WorkerTask result;
    const std::lock_guard<std::mutex> lock(m_queue_lock);

    if (m_workerTaskQueue.size() > 0) {
        result = m_workerTaskQueue.front();
        m_workerTaskQueue.pop_front();
    }

    return result;
}
