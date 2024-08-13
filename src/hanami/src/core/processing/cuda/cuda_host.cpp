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
    initBuffer(localId);
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
 * @brief get next cluster in the queue
 *
 * @return nullptr, if queue is empty, else next cluster in queue
 */
Cluster*
CudaHost::getClusterFromQueue()
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
 * @brief initialize synpase-block-buffer based on the avaialble size of memory
 *
 * @param id local device-id
 */
void
CudaHost::initBuffer(const uint32_t id)
{
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    // m_totalMemory = getAvailableMemory_CUDA(id);
    const uint64_t usedMemory = (m_totalMemory / 100) * 10;  // use 30% for synapse-blocks
    synapseBlocks.initBuffer<SynapseBlock>(usedMemory / sizeof(SynapseBlock));
    synapseBlocks.deleteAll();

    LOG_INFO("Initialized number of syanpse-blocks on gpu-device with id '" + std::to_string(id)
             + "': " + std::to_string(synapseBlocks.metaData->itemCapacity));
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
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    // sync data from gpu to host, in order to have a consistent state
    // see https://github.com/kitsudaiki/Hanami/issues/377
    // copyFromGpu_CUDA(&cluster->gpuPointer,
    //                  cluster->hexagons,
    //                  getItemData<SynapseBlock>(synapseBlocks),
    //                  synapseBlocks.metaData->itemCapacity);

    LogicalHost* originHost = cluster->attachedHost;
    SynapseBlock* cpuSynapseBlocks = Hanami::getItemData<SynapseBlock>(synapseBlocks);
    SynapseBlock tempBlock;

    // copy synapse-blocks from the old host to this one here
    for (uint64_t i = 0; i < cluster->hexagons.size(); i++) {
        for (ConnectionBlock& block : cluster->hexagons[i].connectionBlocks) {
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
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

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
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    // remove synapse-blocks
    for (uint64_t i = 0; i < cluster->hexagons.size(); i++) {
        for (ConnectionBlock& block : cluster->hexagons[i].connectionBlocks) {
            if (block.targetSynapseBlockPos != UNINIT_STATE_64) {
                synapseBlocks.deleteItem(block.targetSynapseBlockPos);
            }
        }
    }

    // remove other data of the cluster, which are no synapse-blocks, from gpu
    // see https://github.com/kitsudaiki/Hanami/issues/377
    // removeFromDevice_CUDA(&cluster->gpuPointer);
}

/**
 * @brief run forward-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CudaHost::trainClusterForward(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    Hanami::ErrorContainer error;

    // see https://github.com/kitsudaiki/Hanami/issues/377
    /* // process input-hexagons
     for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
         Hexagon* hexagon = &cluster->hexagons[hexagonId];
         if (hexagon->isInputHexagon == false) {
             continue;
         }

         processNeuronsOfInputHexagonBackward<true>(
             hexagon, cluster->inputValues, &cluster->neuronBlocks);
     }

     // process all hexagons on cpu
     processing_CUDA(&cluster->gpuPointer,
                     &cluster->hexagons[0],
                     cluster->hexagons.size(),
                     &cluster->neuronBlocks,
                     cluster->numberOfNeuronBlocks,
                     true);

     // process output-hexagons
     for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
         Hexagon* hexagon = &cluster->hexagons[hexagonId];
         if (hexagon->isOutputHexagon == false) {
             continue;
         }
     }

     // update cluster
     if (updateCluster(*cluster)) {
         update_CUDA(&cluster->gpuPointer,
                     &cluster->neuronBlocks,
                     cluster->numberOfNeuronBlocks,
                     &cluster->hexagons[0],
                     cluster->hexagons.size());
     }*/
}

/**
 * @brief run back-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CudaHost::trainClusterBackward(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    Hanami::ErrorContainer error;

    // process output-hexagons on cpu
    for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isOutputHexagon) {
            // see https://github.com/kitsudaiki/Hanami/issues/377
            /*if (backpropagateOutput(&cluster->hexagons[0],
                                    &cluster->outputNeurons[0],
                                    &cluster->neuronBlocks,
                                    &cluster->tempNeuronBlocks,
                                    cluster->outputValues,
                                    cluster->expectedValues,
                                    &cluster->clusterHeader.settings)
                == false)
            {
                return;
            }*/
        }
    }

    // see https://github.com/kitsudaiki/Hanami/issues/377
    // backpropagation over all hexagons on gpu
    /*backpropagation_CUDA(&cluster->gpuPointer,
                         &cluster->hexagons[0],
                         cluster->hexagons.size(),
                         &cluster->neuronBlocks,
                         &cluster->tempNeuronBlocks,
                         cluster->numberOfNeuronBlocks);

    // run reduction-process if enabled
    if (cluster->clusterHeader.settings.enableReduction) {
        if (reductionCounter == 100) {
            reduction_CUDA(&cluster->gpuPointer,
                           &cluster->hexagons[0],
                           cluster->hexagons.size(),
                           &cluster->neuronBlocks,
                           cluster->numberOfNeuronBlocks);
            if (updateCluster(*cluster)) {
                update_CUDA(&cluster->gpuPointer,
                            &cluster->neuronBlocks,
                            cluster->numberOfNeuronBlocks,
                            &cluster->hexagons[0],
                            cluster->hexagons.size());
            }
            reductionCounter = 0;
        }
        reductionCounter++;
    }*/
}

/**
 * @brief process segments
 *
 * @param cluster cluster to process
 */
void
CudaHost::requestCluster(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_cudaMutex);

    Hanami::ErrorContainer error;

    // see https://github.com/kitsudaiki/Hanami/issues/377
    // process input-hexagons
    /*for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isInputHexagon == false) {
            continue;
        }

        processNeuronsOfInputHexagonBackward<false>(
            hexagon, cluster->inputValues, &cluster->neuronBlocks);
    }

    // process all hexagons on gpu
    processing_CUDA(&cluster->gpuPointer,
                    &cluster->hexagons[0],
                    cluster->hexagons.size(),
                    &cluster->neuronBlocks,
                    cluster->numberOfNeuronBlocks,
                    false);*/

    // process output-hexagons
    for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isOutputHexagon == false) {
            continue;
        }
        // see https://github.com/kitsudaiki/Hanami/issues/377
        /*for (uint32_t blockId = 0; blockId < cluster->numberOfNeuronBlocks; ++blockId) {
            processNeuronsOfOutputHexagon(
                hexagon, cluster->outputValues, &cluster->neuronBlocks, blockId);
        }*/
    }
}
