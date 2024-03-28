/**
 * @file        reduction.h
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

#ifndef HANAMI_CORE_REDUCTION_H
#define HANAMI_CORE_REDUCTION_H

#include <common.h>
#include <core/cluster/cluster.h>
#include <core/processing/logical_host.h>
#include <core/processing/objects.h>
#include <hanami_root.h>

/**
 * @brief backpropagate a synapse-section
 *
 * @param section current synapse-section
 */
inline bool
reduceSection(SynapseSection* section)
{
    Synapse* synapse;
    uint8_t exist = 0;

    for (uint8_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            synapse->activeCounter -= static_cast<uint8_t>(synapse->activeCounter < 10);

            // handle active-counter
            if (synapse->activeCounter == 0) {
                if (pos < SYNAPSES_PER_SYNAPSESECTION - 1) {
                    section->synapses[pos] = section->synapses[pos + 1];
                    section->synapses[pos + 1] = Synapse();
                }
                else {
                    section->synapses[pos] = Synapse();
                }
            }
            else {
                exist++;
            }
        }
    }

    // return true;
    return exist != 0;
}

/**
 * @brief backpropagate connections
 *
 * @param brick pointer to current brick
 * @param neuronBlocks pointer to neuron-blocks
 * @param synapseBlocks pointer to synapse-blocks
 * @param blockId id of the current block within the brick
 */
inline void
reduceConnections(Brick* brick,
                  NeuronBlock* neuronBlocks,
                  SynapseBlock* synapseBlocks,
                  const uint32_t blockId)
{
    SynapseConnection* connection = nullptr;
    Neuron* sourceNeuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* synapseSection = nullptr;
    const uint32_t dimY = brick->dimY;
    const uint32_t dimX = brick->dimX;

    if (blockId >= dimX) {
        return;
    }

    for (uint32_t c = blockId * dimY; c < (blockId * dimY) + dimY; ++c) {
        connectionBlock = &brick->connectionBlocks[c];

        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            connection = &connectionBlock->connections[i];
            if (connection->origin.blockId == UNINIT_STATE_32) {
                continue;
            }

            synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];
            sourceNeuronBlock = &neuronBlocks[connection->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[connection->origin.neuronId];

            // if section is complete empty, then erase it
            if (reduceSection(synapseSection) == false) {
                // initialize the creation of a new section
                sourceNeuron->isNew = 1;
                sourceNeuron->newLowerBound = connection->lowerBound;
                sourceNeuron->inUse &= (~(1 << connection->origin.posInNeuron));

                // mark current connection as available again
                connection->origin.blockId = UNINIT_STATE_32;
                connection->origin.neuronId = UNINIT_STATE_16;
                connection->origin.posInNeuron = 0;
            }
        }
    }
}

/**
 * @brief reduce all synapses within the cluster and delete them, if the reach a deletion-border
 */
inline void
reduceCluster(const Cluster& cluster, const uint32_t brickId, const uint32_t blockId)
{
    Brick* brick = &cluster.bricks[brickId];
    if (brick->isInputBrick) {
        return;
    }

    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    reduceConnections(brick, cluster.neuronBlocks, synapseBlocks, blockId);
}

#endif  // HANAMI_CORE_REDUCTION_H
