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
                     TempNeuronBlock* targetTempBlock,
                     Neuron* sourceNeuron,
                     TempNeuron* sourceTempNeuron)
{
    float counter = sourceNeuron->potential - connection->offset;
    uint pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    TempNeuron* targetTempNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    float delta = 0.0f;
    sourceTempNeuron->delta[connection->origin.posInNeuron] = 0.0f;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            targetTempNeuron = &targetTempBlock->neurons[synapse->targetNeuronId];

            delta = targetTempNeuron->delta[0] * synapse->weight;
            if (counter < synapse->border) {
                delta *= (1.0f / synapse->border) * counter;
            }
            synapse->weight -= trainValue * targetTempNeuron->delta[0];
            sourceTempNeuron->delta[connection->origin.posInNeuron] += delta;
            // if(connection->origin.posInNeuron == 2) {
            //     printf("poi %d \n", connection->origin.posInNeuron);
            // }
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
backpropagateNeurons(Brick* brick,
                     NeuronBlock* neuronBlocks,
                     TempNeuronBlock* tempNeuronBlocks,
                     SynapseBlock* synapseBlocks)
{
    SynapseConnection* scon = nullptr;

    Neuron* sourceNeuron = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;

    TempNeuron* sourceTempNeuron = nullptr;
    TempNeuron* targetTempNeuron = nullptr;
    TempNeuronBlock* sourceTempBlock = nullptr;
    TempNeuronBlock* targetTempBlock = nullptr;

    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* section = nullptr;
    float delta;

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->neuronBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         ++blockId)
    {
        targetNeuronBlock = &neuronBlocks[blockId];
        targetTempBlock = &tempNeuronBlocks[blockId];

        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < NEURONS_PER_NEURONSECTION;
             ++neuronIdInBlock)
        {
            targetNeuron = &targetNeuronBlock->neurons[neuronIdInBlock];
            targetTempNeuron = &targetTempBlock->neurons[neuronIdInBlock];

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

    for (uint32_t c = 0; c < brick->connectionBlocks.size(); c++) {
        connectionBlock = &brick->connectionBlocks[c];

        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            scon = &connectionBlock->connections[i];
            if (scon->origin.blockId == UNINIT_STATE_32) {
                continue;
            }

            section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];

            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceTempBlock = &tempNeuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.sectionId];
            sourceTempNeuron = &sourceTempBlock->neurons[scon->origin.sectionId];

            targetTempBlock = &tempNeuronBlocks[brick->neuronBlockPos + (c / brick->dimY)];

            backpropagateSection(section, scon, targetTempBlock, sourceNeuron, sourceTempNeuron);
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
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::cpuSynapseBlocks);

    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId) {
        Brick* brick = &cluster.bricks[brickId];
        if (brick->isOutputBrick) {
            if (backpropagateOutput(brick,
                                    cluster.neuronBlocks,
                                    cluster.tempNeuronBlocks,
                                    outputValues,
                                    expectedValues,
                                    &cluster.clusterHeader->settings)
                == false)
            {
                return;
            }
        }

        if (brick->isInputBrick == false) {
            backpropagateNeurons(
                brick, cluster.neuronBlocks, cluster.tempNeuronBlocks, synapseBlocks);
        }
    }
}

#endif  // HANAMI_CORE_BACKPROPAGATION_H
