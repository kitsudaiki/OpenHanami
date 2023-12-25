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
 * @brief backpropagateSection
 * @param section
 * @param connection
 * @param targetNeuronBlock
 * @param sourceNeuron
 */
void
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     NeuronBlock* targetNeuronBlock,
                     Neuron* sourceNeuron)
{
    float counter = sourceNeuron->potential - connection->offset;
    uint pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    float delta = 0.0f;
    sourceNeuron->delta[connection->origin.posInNeuron] = 0.0f;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        // break look, if no more synapses to process
        synapse = &section->synapses[pos];
        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            // update weight
            targetNeuron = &targetNeuronBlock->neurons[synapse->targetNeuronId];
            delta = targetNeuron->delta[0] * synapse->weight;
            if (counter < synapse->border) {
                delta *= (1.0f / synapse->border) * counter;
            }
            synapse->weight -= trainValue * targetNeuron->delta[0];
            sourceNeuron->delta[connection->origin.posInNeuron] += delta;

            counter -= synapse->border;
        }
        ++pos;
    }
}

/**
 * @brief backpropagateNeurons
 * @param brick
 * @param neuronBlocks
 * @param synapseBlocks
 */
inline void
backpropagateNeurons(Brick* brick, NeuronBlock* neuronBlocks, SynapseBlock* synapseBlocks)
{
    SynapseConnection* scon = nullptr;
    Neuron* neuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* section = nullptr;
    float delta;

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->neuronBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         ++blockId)
    {
        targetNeuronBlock = &neuronBlocks[blockId];
        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < NEURONS_PER_NEURONSECTION;
             ++neuronIdInBlock)
        {
            neuron = &targetNeuronBlock->neurons[neuronIdInBlock];
            if (neuron->active == false) {
                continue;
            }

            // aggregate different delta-values
            delta = 0.0f;
            delta += neuron->delta[0];
            delta += neuron->delta[1];
            delta += neuron->delta[2];
            delta += neuron->delta[3];
            delta += neuron->delta[4];
            delta += neuron->delta[5];
            delta += neuron->delta[6];
            delta += neuron->delta[7];

            // calculate new delta-value for next iteration
            delta *= 1.4427f * pow(0.5f, neuron->potential);
            neuron->delta[0] = delta;
            neuron->delta[1] = 0.0f;
            neuron->delta[2] = 0.0f;
            neuron->delta[3] = 0.0f;
            neuron->delta[4] = 0.0f;
            neuron->delta[5] = 0.0f;
            neuron->delta[6] = 0.0f;
            neuron->delta[7] = 0.0f;
        }
    }

    for (uint32_t c = 0; c < brick->connectionBlocks.size(); c++) {
        connectionBlock = &brick->connectionBlocks[c];
        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            scon = &connectionBlock->connections[i];
            if (scon->origin.blockId == UNINIT_STATE_32) {
                continue;
            }
            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.sectionId];
            section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];
            targetNeuronBlock = &neuronBlocks[brick->neuronBlockPos + (c / brick->dimY)];

            backpropagateSection(section, scon, targetNeuronBlock, sourceNeuron);
        }
    }
}

/**
 * @brief reweightCoreSegment
 * @param cluster
 */
void
reweightCoreSegment(const Cluster& cluster)
{
    float* expectedValues = cluster.expectedValues;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::cpuSynapseBlocks);

    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId) {
        Brick* brick = &cluster.bricks[brickId];
        if (brick->isOutputBrick) {
            if (backpropagateOutput(brick,
                                    neuronBlocks,
                                    outputValues,
                                    expectedValues,
                                    &cluster.clusterHeader->settings)
                == false)
            {
                return;
            }
        }

        if (brick->isInputBrick == false) {
            backpropagateNeurons(brick, neuronBlocks, synapseBlocks);
        }
    }
}

#endif  // HANAMI_CORE_BACKPROPAGATION_H
