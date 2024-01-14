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
#include <hanami_cpu/memory.h>

CpuHost::CpuHost(const uint32_t localId) : LogicalHost(localId)
{
    m_hostType = CPU_HOST_TYPE;
    initBuffer(localId);
}

uint64_t
CpuHost::getAvailableMemory()
{
    return 0;
}

/**
 * @brief LogicalHost::moveCluster
 * @param originHost
 * @param cluster
 * @return
 */
bool
CpuHost::moveCluster(Cluster* cluster)
{
    LogicalHost* originHost = cluster->attachedHost;
    SynapseBlock* cpuSynapseBlocks = Hanami::getItemData<SynapseBlock>(synapseBlocks);
    SynapseBlock tempBlock;

    for (uint64_t i = 0; i < cluster->clusterHeader->bricks.count; i++) {
        for (ConnectionBlock& block : cluster->bricks[i].connectionBlocks) {
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

void
CpuHost::syncWithHost(Cluster*)
{
}

void
CpuHost::removeCluster(Cluster* cluster)
{
    SynapseBlock* cpuSynapseBlocks = Hanami::getItemData<SynapseBlock>(synapseBlocks);
    SynapseBlock tempBlock;

    for (uint64_t i = 0; i < cluster->clusterHeader->bricks.count; i++) {
        for (ConnectionBlock& block : cluster->bricks[i].connectionBlocks) {
            if (block.targetSynapseBlockPos != UNINIT_STATE_64) {
                synapseBlocks.deleteItem(block.targetSynapseBlockPos);
            }
        }
    }
}

/**
 * @brief run forward-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CpuHost::trainClusterForward(Cluster* cluster)
{
    Hanami::ErrorContainer error;

    processCluster(*cluster, true);
}

/**
 * @brief run back-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CpuHost::trainClusterBackward(Cluster* cluster)
{
    Hanami::ErrorContainer error;

    reweightCluster(*cluster);

    if (reductionCounter == 100) {
        reduceCluster(*cluster);
        updateCluster(*cluster);
        reductionCounter = 0;
    }
    reductionCounter++;
}

/**
 * @brief process segments
 *
 * @param cluster cluster to process
 */
void
CpuHost::requestCluster(Cluster* cluster)
{
    Hanami::ErrorContainer error;
    processCluster(*cluster, false);
}

void
CpuHost::initBuffer(const uint32_t id)
{
    uint64_t sizeOfMemory = getFreeMemory();
    sizeOfMemory = (sizeOfMemory / 100) * 80;  // use 80% for synapse-blocks
    synapseBlocks.initBuffer<SynapseBlock>(sizeOfMemory / sizeof(SynapseBlock));
    synapseBlocks.deleteAll();

    LOG_INFO("Initialized number of syanpse-blocks on cpu-device with id '" + std::to_string(id)
             + "': " + std::to_string(synapseBlocks.metaData->itemCapacity));
}
