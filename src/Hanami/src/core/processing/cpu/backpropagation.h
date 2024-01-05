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

#include <common.h>
#include <core/cluster/cluster.h>
#include <core/processing/cluster_io_functions.h>
#include <core/processing/objects.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief backpropagate all neurons
 *
 * @param brick pointer to current brick
 * @param neuronBlocks pointer to neuron-blocks
 * @param synapseBlocks pointer to synapse-blocks
 */
inline void
backpropagateNeuron(Brick* brick, NeuronBlock* neuronBlocks, TempNeuronBlock* tempNeuronBlocks)
{
    SynapseConnection* scon = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    TempNeuron* targetTempNeuron = nullptr;
    TempNeuronBlock* targetTempBlock = nullptr;
    float delta = 0.0f;

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->neuronBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         ++blockId)
    {
        targetNeuronBlock = &neuronBlocks[blockId];
        targetTempBlock = &tempNeuronBlocks[blockId];

        for (uint32_t neuronId = 0; neuronId < NEURONS_PER_NEURONSECTION; ++neuronId) {
            targetNeuron = &targetNeuronBlock->neurons[neuronId];
            targetTempNeuron = &targetTempBlock->neurons[neuronId];

            if (targetNeuron->active == false) {
                continue;
            }

            // aggregate different delta-values
            delta = 0.0f;
            delta += targetTempNeuron->delta[0];
            delta += targetTempNeuron->delta[1];
            delta += targetTempNeuron->delta[2];
            delta += targetTempNeuron->delta[3];
            delta += targetTempNeuron->delta[4];
            delta += targetTempNeuron->delta[5];
            delta += targetTempNeuron->delta[6];
            delta += targetTempNeuron->delta[7];

            // calculate new delta-value for next iteration
            delta *= 1.4427f * pow(0.5f, targetNeuron->potential);
            targetTempNeuron->delta[0] = delta;
        }
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
void
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     NeuronBlock* targetBlock,
                     TempNeuronBlock* targetTempBlock,
                     Neuron* sourceNeuron,
                     TempNeuron* sourceTempNeuron)
{
    float potential = sourceNeuron->potential - connection->offset;
    uint8_t pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    TempNeuron* targetTempNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    float delta = 0.0f;
    sourceTempNeuron->delta[connection->origin.posInNeuron] = 0.0f;
    uint8_t active = 0;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && potential > 0.01f) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            targetTempNeuron = &targetTempBlock->neurons[synapse->targetNeuronId];
            targetNeuron = &targetBlock->neurons[synapse->targetNeuronId];
            active = (targetNeuron->potential > 0.0f) == (synapse->weight > 0.0f);
            synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 10);

            delta = targetTempNeuron->delta[0] * synapse->weight;
            if (potential < synapse->border) {
                delta *= (1.0f / synapse->border) * potential;
            }
            synapse->weight -= trainValue * targetTempNeuron->delta[0];
            sourceTempNeuron->delta[connection->origin.posInNeuron] += delta;

            potential -= synapse->border;
        }
        ++pos;
    }
}

/**
 * @brief backpropagate connections
 *
 * @param brick pointer to current brick
 * @param neuronBlocks pointer to neuron-blocks
 * @param synapseBlocks pointer to synapse-blocks
 */
inline void
backpropagateConnections(Brick* brick,
                         NeuronBlock* neuronBlocks,
                         TempNeuronBlock* tempNeuronBlocks,
                         SynapseBlock* synapseBlocks)
{
    SynapseConnection* connection = nullptr;
    Neuron* sourceNeuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    TempNeuron* sourceTempNeuron = nullptr;
    TempNeuronBlock* sourceTempBlock = nullptr;
    TempNeuronBlock* targetTempBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* synapseSection = nullptr;

    for (uint32_t c = 0; c < brick->connectionBlocks.size(); c++) {
        connectionBlock = &brick->connectionBlocks[c];

        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            connection = &connectionBlock->connections[i];

            if (connection->origin.blockId == UNINIT_STATE_32) {
                continue;
            }

            synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];

            sourceNeuronBlock = &neuronBlocks[connection->origin.blockId];
            sourceTempBlock = &tempNeuronBlocks[connection->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[connection->origin.neuronId];
            sourceTempNeuron = &sourceTempBlock->neurons[connection->origin.neuronId];

            targetTempBlock = &tempNeuronBlocks[brick->neuronBlockPos + (c / brick->dimY)];
            targetNeuronBlock = &neuronBlocks[brick->neuronBlockPos + (c / brick->dimY)];

            backpropagateSection(synapseSection,
                                 connection,
                                 targetNeuronBlock,
                                 targetTempBlock,
                                 sourceNeuron,
                                 sourceTempNeuron);
        }
    }
}

/**
 * @brief adjust weights of synapses within a cluster
 *
 * @param cluster cluster, which should be updated
 */
void
reweightCluster(const Cluster& cluster)
{
    Brick* brick = nullptr;
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::cpuSynapseBlocks);
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;

    // backpropagate all output-values
    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isOutputBrick) {
            if (backpropagateOutput(brick,
                                    cluster.neuronBlocks,
                                    cluster.tempNeuronBlocks,
                                    cluster.outputValues,
                                    cluster.expectedValues,
                                    &cluster.clusterHeader->settings)
                == false)
            {
                return;
            }
        }
    }

    // process normal and output-bricks
    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isInputBrick == false) {
            backpropagateNeuron(brick, cluster.neuronBlocks, cluster.tempNeuronBlocks);
            backpropagateConnections(
                brick, cluster.neuronBlocks, cluster.tempNeuronBlocks, synapseBlocks);
        }
    }
}

#endif  // HANAMI_CORE_BACKPROPAGATION_H
