/**
 * @file        backpropagation.h
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

#ifndef HANAMI_CORE_BACKPROPAGATION_H
#define HANAMI_CORE_BACKPROPAGATION_H

#include <core/cluster/cluster.h>
#include <core/cluster/objects.h>
#include <core/processing/cluster_io_functions.h>
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/logical_host.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief backpropagate all neurons
 *
 * @param brick pointer to current brick
 * @param neuronBlocks pointer to neuron-blocks
 * @param synapseBlocks pointer to synapse-blocks
 * @param blockId id of the current block within the brick
 */
inline void
backpropagateNeuron(Brick* brick, const uint32_t blockId)
{
    SynapseConnection* scon = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    float delta = 0.0f;
    const uint32_t neuronBlockId = blockId;
    uint8_t neuronId = 0;

    targetNeuronBlock = &brick->neuronBlocks[neuronBlockId];

    for (neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
        targetNeuron = &targetNeuronBlock->neurons[neuronId];

        if (targetNeuron->active == false) {
            continue;
        }
        targetNeuron->delta *= 1.4427f * pow(0.5f, targetNeuron->potential);
    }
}

/**
 * @brief backpropagate a synapse-section
 *
 * @param section current synapse-section
 * @param connection current connection related to the synapse-section
 * @param targetTempBlock temp-value-block of the target neuron-block
 * @param sourceNeuron source-neuron, which triggered the section
 * @param sourceTempNeuron temp-balue block of the source-neuron
 */
inline void
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     NeuronBlock* targetBlock,
                     Neuron* sourceNeuron)
{
    float potential = sourceNeuron->potential - connection->lowerBound;
    uint8_t pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    float delta = 0.0f;
    uint8_t active = 0;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && potential > 0.00001f) {
        synapse = &section->synapses[pos];
        ++pos;

        if (synapse->targetNeuronId == UNINIT_STATE_8) {
            continue;
        }

        targetNeuron = &targetBlock->neurons[synapse->targetNeuronId];
        active = (targetNeuron->potential > 0.0f) == (synapse->weight > 0.0f);
        synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 10);

        delta = targetNeuron->delta * synapse->weight;
        synapse->weight -= trainValue * targetNeuron->delta;
        sourceNeuron->delta += delta;

        potential -= synapse->border;
    }
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
backpropagateConnections(Brick* brick,
                         Brick* bricks,
                         SynapseBlock* synapseBlocks,
                         const uint32_t blockId)
{
    SynapseConnection* connection = nullptr;
    Neuron* sourceNeuron = nullptr;
    Brick* sourceBrick = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* synapseSection = nullptr;
    const uint32_t dimY = brick->header.dimY;
    const uint32_t dimX = brick->header.dimX;
    SourceLocation sourceLoc;
    uint32_t c = 0;
    uint32_t i = 0;

    if (blockId >= dimX) {
        return;
    }

    for (c = blockId * dimY; c < (blockId * dimY) + dimY; ++c) {
        assert(c < brick->connectionBlocks.size());
        connectionBlock = &brick->connectionBlocks[c];

        for (i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
            connection = &connectionBlock->connections[i];

            if (connection->origin.blockId == UNINIT_STATE_16) {
                continue;
            }

            synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];
            sourceLoc = getSourceNeuron(connection->origin, bricks);

            targetNeuronBlock = &brick->neuronBlocks[(c / brick->header.dimY)];

            backpropagateSection(synapseSection, connection, targetNeuronBlock, sourceLoc.neuron);
        }
    }
}

#endif  // HANAMI_CORE_BACKPROPAGATION_H
