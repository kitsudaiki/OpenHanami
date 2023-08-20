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
#include <math.h>
#include <cmath>

#include <hanami_root.h>
#include <core/processing/objects.h>
#include <core/cluster/cluster.h>
#include <core/processing/cluster_io_functions.h>

/**
 * @brief run backpropagation for a single synapse-section
 */
inline void
backpropagateSection(const Cluster &cluster,
                     Synapse* section,
                     Neuron* sourceNeuron,
                     SynapseConnection* connection,
                     LocationPtr* sourceLocation,
                     const float outH,
                     NeuronBlock* neuronBlocks,
                     SynapseBlock* synapseBlocks,
                     SynapseConnection* synapseConnections)
{
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    float learnValue = 0.2f;
    uint16_t pos = 0;
    float counter = outH - connection->offset[sourceLocation->sectionId];
    NeuronBlock* neuronBlock = &neuronBlocks[connection->targetNeuronBlockId];

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && counter > 0.01f)
    {
        // break look, if no more synapses to process
        synapse = &section[pos];
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            pos++;
            continue;
        }

        // update weight
        learnValue = static_cast<float>(126 - synapse->activeCounter) * 0.0002f;
        learnValue += 0.05f;
        targetNeuron = &neuronBlock->neurons[synapse->targetNeuronId];
        sourceNeuron->delta += targetNeuron->delta * synapse->weight;
        synapse->weight -= learnValue * targetNeuron->delta;

        counter -= synapse->border;
        pos++;
    }

    LocationPtr* targetLocation = &connection->next[sourceLocation->sectionId];
    if(targetLocation->sectionId != UNINIT_STATE_16)
    {
        Synapse* nextSection = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
        SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
        backpropagateSection(cluster,
                             nextSection,
                             sourceNeuron,
                             nextConnection,
                             targetLocation,
                             outH,
                             neuronBlocks,
                             synapseBlocks,
                             synapseConnections);
    }
}

/**
 * @brief run back-propagation over all neurons
 */
inline void
backpropagateNeurons(const Cluster &cluster,
                     const Brick* brick,
                     NeuronBlock* neuronBlocks,
                     SynapseBlock* synapseBlocks,
                     SynapseConnection* synapseConnections)
{
    Neuron* sourceNeuron = nullptr;
    NeuronBlock* block = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
        blockId++)
    {
        block = &neuronBlocks[blockId];
        for(uint32_t neuronIdInBlock = 0;
            neuronIdInBlock < block->numberOfNeurons;
            neuronIdInBlock++)
        {
            // skip section, if not active
            sourceNeuron = &block->neurons[neuronIdInBlock];
            if(sourceNeuron->target.blockId == UNINIT_STATE_32) {
                continue;
            }

            // set start-values
            sourceNeuron->delta = 0.0f;
            if(sourceNeuron->active)
            {
                LocationPtr* targetLocation = &sourceNeuron->target;
                if(targetLocation->blockId == UNINIT_STATE_32) {
                    return;
                }

                Synapse* nextSection = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
                SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
                backpropagateSection(cluster,
                                     nextSection,
                                     sourceNeuron,
                                     nextConnection,
                                     targetLocation,
                                     sourceNeuron->potential,
                                     neuronBlocks,
                                     synapseBlocks,
                                     synapseConnections);

                sourceNeuron->delta *= 1.4427f * pow(0.5f, sourceNeuron->potential);
            }
        }
    }
}

/**
 * @brief correct weight of synapses within a segment
 */
void
reweightCoreSegment(const Cluster &cluster)
{
    float* expectedValues = cluster.expectedValues;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseConnection* synapseConnections = cluster.synapseConnections;
    SynapseBlock* synapseBlocks = cluster.synapseBlocks;

    // run back-propagation over all internal neurons and synapses
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for(int32_t pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        const uint32_t brickId = cluster.brickOrder[pos];
        Brick* brick = &cluster.bricks[brickId];
        if(brick->isOutputBrick)
        {
            if(backpropagateOutput(brick,
                                   neuronBlocks,
                                   outputValues,
                                   expectedValues,
                                   cluster.clusterSettings) == false)
            {
                return;
            }
        }
        backpropagateNeurons(cluster,
                             brick,
                             neuronBlocks,
                             synapseBlocks,
                             synapseConnections);
    }
}

#endif // HANAMI_CORE_BACKPROPAGATION_H
