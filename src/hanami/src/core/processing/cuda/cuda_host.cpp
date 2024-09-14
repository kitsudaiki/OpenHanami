/**
 * @file        cuda_host.cpp
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

#include "cuda_host.h"

#include <core/processing/cluster_resize.h>
#include <core/processing/cuda/cuda_functions.h>

/**
 * @brief constructor
 *
 * @param localId identifier starting with 0 within the physical host and with the type of host
 */
CudaHost::CudaHost(const uint32_t localId) : LogicalHost(localId)
{
    m_hostType = CUDA_HOST_TYPE;

    initBuffer();
    initWorkerThreads();
}

/**
 * @brief destructor
 */
CudaHost::~CudaHost() {}

/**
 * @brief add cluster to queue
 *
 * @param cluster cluster to add to queue
 */
void
CudaHost::addClusterToHost(Cluster* cluster)
{
    while (m_queue_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }
    m_clusterQueue.push_back(cluster);
    m_queue_lock.clear(std::memory_order_release);
}

/**
 * @brief initialize synpase-block-buffer based on the avaialble size of memory
 *
 * @param id local device-id
 */
void
CudaHost::initBuffer()
{
    const std::lock_guard<std::mutex> lock(cudaMutex);

    // m_totalMemory = getAvailableMemory_CUDA(id);
    const uint64_t usedMemory = (m_totalMemory / 100) * 10;  // use 30% for synapse-blocks
    synapseBlocks.initBuffer<SynapseBlock>(usedMemory / sizeof(SynapseBlock));
    synapseBlocks.deleteAll();

    LOG_INFO("Initialized number of syanpse-blocks on gpu-device: "
             + std::to_string(synapseBlocks.metaData->itemCapacity));
}

/**
 * @brief move the data of a cluster to this host
 *
 * @param cluster cluster to move
 *
 * @return true, if successful, else false
 */
bool
CudaHost::moveCluster(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(cudaMutex);

    // sync data from gpu to host, in order to have a consistent state
    // see https://github.com/kitsudaiki/Hanami/issues/377
    // copyFromGpu_CUDA(&cluster->gpuPointer,
    //                  cluster->hexagons,
    //                  getItemData<SynapseBlock>(synapseBlocks),
    //                  synapseBlocks.metaData->itemCapacity);

    LogicalHost* originHost = cluster->attachedHost;
    SynapseBlock* cpuSynapseBlocks = Hanami::getItemData<SynapseBlock>(synapseBlocks);
    SynapseBlock tempBlock;

    // copy synapse-blocks from the old host to this one
    for (uint64_t i = 0; i < cluster->hexagons.size(); i++) {
        for (uint64_t pos = 0; pos < cluster->hexagons[i].synapseBlockLinks.size(); pos++) {
            const uint64_t synapseSectionPos = cluster->hexagons[i].synapseBlockLinks[pos];
            if (synapseSectionPos != UNINIT_STATE_64) {
                tempBlock = cpuSynapseBlocks[synapseSectionPos];
                originHost->synapseBlocks.deleteItem(synapseSectionPos);
                const uint64_t newPos = synapseBlocks.addNewItem(tempBlock);
                // TODO: make roll-back possible in error-case
                if (newPos == UNINIT_STATE_64) {
                    return false;
                }
                cluster->hexagons[i].synapseBlockLinks[pos] = newPos;
            }
        }
    }

    // update data on gpu
    cluster->gpuPointer.deviceId = m_localId;
    // see https://github.com/kitsudaiki/Hanami/issues/377
    // copyToDevice_CUDA(&cluster->gpuPointer,
    //                   &cluster->clusterHeader.settings,
    //                   cluster->hexagons,
    //                   getItemData<SynapseBlock>(synapseBlocks),
    //                   synapseBlocks.metaData->itemCapacity);

    cluster->attachedHost = this;

    return true;
}
/**
 * @brief sync data of a cluster from gpu to host
 *
 * @param cluster cluster to sync
 */
void
CudaHost::syncWithHost(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(cudaMutex);

    // see https://github.com/kitsudaiki/Hanami/issues/377
    // copyFromGpu_CUDA(&cluster->gpuPointer,
    //                  cluster->hexagons,
    //                  getItemData<SynapseBlock>(synapseBlocks),
    //                  synapseBlocks.metaData->itemCapacity);
}

/**
 * @brief remove the cluster-data from this host
 *
 * @param cluster cluster to remove
 */
void
CudaHost::removeCluster(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(cudaMutex);

    // remove synapse-blocks
    for (uint64_t i = 0; i < cluster->hexagons.size(); i++) {
        for (uint64_t& link : cluster->hexagons[i].synapseBlockLinks) {
            if (link != UNINIT_STATE_64) {
                synapseBlocks.deleteItem(link);
            }
        }
    }

    // remove other data of the cluster, which are no synapse-blocks, from gpu
    // see https://github.com/kitsudaiki/Hanami/issues/377
    // removeFromDevice_CUDA(&cluster->gpuPointer);
}

bool
CudaHost::initWorkerThreads()
{
    return true;
}
