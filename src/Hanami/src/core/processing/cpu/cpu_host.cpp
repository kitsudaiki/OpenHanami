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

#include <core/processing/cluster_io_functions.h>
#include <core/processing/cluster_resize.h>
#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/cpu/worker_thread.h>
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
    if (cluster->mode == ClusterProcessingMode::REDUCTION_MODE) {
        addBrickToTaskQueue(cluster, 0);
    }
    else {
        addBrickToTaskQueue(cluster, UNINIT_STATE_32);
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
 * @brief add a brick to the task-queue, which is used source source for the worker-threads
 *
 * @param cluster related cluster
 * @param brickId brick-id to process
 */
void
CpuHost::addBrickToTaskQueue(Cluster* cluster, const u_int32_t brickId)
{
    if (brickId == UNINIT_STATE_32) {
        // special case, where based on the cluster-mode, the whole firsth or last
        // brick should be processed by a single worker-threads
        WorkerTask task;
        task.cluster = cluster;
        task.brickId = brickId;
        addWorkerTaskToQueue(task);
    }
    else {
        for (uint32_t i = 0; i < cluster->bricks[brickId].numberOfNeuronBlocks; i++) {
            WorkerTask task;
            task.cluster = cluster;
            task.brickId = brickId;
            task.blockId = i;
            addWorkerTaskToQueue(task);
        }
    }
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
    const uint64_t usedMemory = (m_totalMemory / 100) * 80;  // use 80% for synapse-blocks
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
    for (Brick& brick : cluster->bricks) {
        for (ConnectionBlock& block : brick.connectionBlocks[0]) {
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
    for (Brick& brick : cluster->bricks) {
        for (ConnectionBlock& block : brick.connectionBlocks[0]) {
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
