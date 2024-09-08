﻿/**
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
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/logical_host.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief backpropagate all neurons
 *
 * @param hexagon pointer to current hexagon
 * @param blockId id of the current block within the hexagon
 */
inline void
backpropagateNeuron(Hexagon* hexagon, const uint32_t blockId)
{
    Connection* scon = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    float delta = 0.0f;
    const uint32_t neuronBlockId = blockId;
    uint8_t neuronId = 0;

    targetNeuronBlock = &hexagon->neuronBlocks[neuronBlockId];

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
 */
inline void
backpropagateSection(SynapseSection* section,
                     Connection* connection,
                     NeuronBlock* targetBlock,
                     Neuron* sourceNeuron)
{
    float potential = sourceNeuron->potential - connection->lowerBound;
    uint8_t pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    constexpr float trainValue = 0.1f;
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
 * @param hexagon pointer to current hexagon
 * @param hexagons pointer to list of all hexagons
 * @param synapseBlocks pointer to synapse-blocks
 * @param blockId id of the current block within the hexagon
 */
inline void
backpropagateConnections(Hexagon* hexagon,
                         Hexagon* hexagons,
                         SynapseBlock* synapseBlocks,
                         const uint32_t blockId)
{
    Connection* connection = nullptr;
    Neuron* sourceNeuron = nullptr;
    Hexagon* sourceHexagon = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* synapseSection = nullptr;
    const uint32_t dimX = hexagon->header.dimX;
    SourceLocation sourceLoc;

    if (blockId >= dimX) {
        return;
    }

    assert(blockId < hexagon->connectionBlocks.size());
    connectionBlock = &hexagon->connectionBlocks[blockId];
    const uint64_t synapseBlockLink = hexagon->synapseBlockLinks[blockId];

    for (uint32_t i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
        connection = &connectionBlock->connections[i];

        if (connection->origin.blockId == UNINIT_STATE_16) {
            continue;
        }

        synapseSection = &synapseBlocks[synapseBlockLink].sections[i];
        sourceLoc = getSourceNeuron(connection->origin, hexagons);

        targetNeuronBlock = &hexagon->neuronBlocks[blockId];

        backpropagateSection(synapseSection, connection, targetNeuronBlock, sourceLoc.neuron);
    }
}

// Derivative of the activation function
inline float
sigmoidDerivative(const float x)
{
    return x * (1 - x);
}

/**
 * @brief backpropagate output-nodes
 *
 * @param hexagons list of all hexagons
 * @param outputInterface pointer ot the connected output-interface
 * @param settings pointer cluster-settings
 * @param hexagonId current hexagon-id
 *
 * @return always true
 */
inline bool
backpropagateOutput(std::vector<Hexagon>& hexagons,
                    OutputInterface* outputInterface,
                    const ClusterSettings* settings,
                    const uint32_t hexagonId)
{
    Neuron* neuron = nullptr;
    Hexagon* hexagon = nullptr;
    OutputNeuron* out = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    float totalDelta = 0.0f;
    float learnValue = 0.1f;
    float delta = 0.0f;
    float update = 0.0f;
    uint64_t i = 0;
    uint64_t j = 0;

    for (i = 0; i < outputInterface->outputNeurons.size(); ++i) {
        out = &outputInterface->outputNeurons[i];
        hexagon = &hexagons[outputInterface->targetHexagonId];
        delta = out->outputVal - out->exprectedVal;
        update = delta * sigmoidDerivative(out->outputVal);

        for (j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if (target->blockId == UNINIT_STATE_16) {
                continue;
            }

            neuron = &hexagon->neuronBlocks[target->blockId].neurons[target->neuronId];
            neuron->delta += update * target->connectionWeight;
            target->connectionWeight -= update * learnValue * neuron->potential;

            totalDelta += abs(delta);
        }
    }

    hexagon = &hexagons[hexagonId];
    for (i = 0; i < hexagon->neuronBlocks.size(); ++i) {
        for (j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &hexagon->neuronBlocks[i].neurons[j];
            neuron->delta *= sigmoidDerivative(neuron->potential);
        }
    }

    return true;
    // return totalDelta > settings->backpropagationBorder;
}

/**
 * @brief run the backpropagation over the core the cluster
 *
 * @param cluster pointer to cluster to process
 * @param hexagonId id of the hexagon to process
 * @param blockId id of the block within the hexagon
 */
inline void
processClusterBackward(Cluster& cluster, const uint32_t hexagonId, const uint32_t blockId)
{
    Hanami::ErrorContainer error;
    Hexagon* hexagon = &cluster.hexagons[hexagonId];
    if (hexagon->header.isInputHexagon) {
        return;
    }

    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    backpropagateNeuron(hexagon, blockId);
    backpropagateConnections(hexagon, &cluster.hexagons[0], synapseBlocks, blockId);
}

#endif  // HANAMI_CORE_BACKPROPAGATION_H
