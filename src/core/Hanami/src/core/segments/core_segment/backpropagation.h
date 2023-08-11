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
#include <core/segments/brick.h>
#include <core/segments/core_segment/core_segment.h>

#include "objects.h"

/**
 * @brief backpropagate values of an output-brick
 */
inline bool
backpropagateOutput(const CoreSegment &segment,
                    const Brick* brick)
{
    Neuron* neuron = nullptr;
    BrickBlock* section = nullptr;
    float totalDelta = 0.0f;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->brickBlockPos;
        neuronSectionId < brick->numberOfNeuronSections + brick->brickBlockPos;
        neuronSectionId++)
    {
        section = &segment.brickBlocks[neuronSectionId];
        for(uint32_t neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            neuron->delta = segment.inputTransfers[neuron->targetBorderId];
            segment.inputTransfers[neuron->targetBorderId] = 0.0f;
            totalDelta += abs(neuron->delta);
        }
    }

    return totalDelta > segment.segmentSettings->backpropagationBorder;
    //return true;
}

/**
 * @brief run backpropagation for a single synapse-section
 */
inline void
backpropagateSection(const CoreSegment &segment,
                     Synapse* section,
                     Neuron* sourceNeuron,
                     BrickBlock* block,
                     BlockConnection* connection,
                     LocationPtr* sourceLocation,
                     const float outH)
{
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    float learnValue = 0.2f;
    uint16_t pos = 0;
    float counter = outH - connection->offset[sourceLocation->sectionId];

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
        targetNeuron = &block->neurons[synapse->targetNeuronId];
        sourceNeuron->delta += targetNeuron->delta * synapse->weight;
        synapse->weight -= learnValue * targetNeuron->delta;

        counter -= synapse->border;
        pos++;
    }

    LocationPtr* targetLocation = &connection->next[sourceLocation->sectionId + 64];
    if(targetLocation->sectionId != UNINIT_STATE_16)
    {
        Synapse* nextSection = block->synapseBlock.synapses[targetLocation->sectionId];
        BrickBlock* nextBlock = &segment.brickBlocks[targetLocation->blockId];
        BlockConnection* nextConnection = &segment.blockConnections[targetLocation->blockId];
        backpropagateSection(segment,
                             nextSection,
                             sourceNeuron,
                             nextBlock,
                             nextConnection,
                             targetLocation,
                             outH);
    }
}

/**
 * @brief run back-propagation over all neurons
 */
inline void
backpropagateNeurons(const CoreSegment &segment,
                     const Brick* brick)
{
    Neuron* sourceNeuron = nullptr;
    BrickBlock* block = nullptr;
    BlockConnection* connection = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronSections + brick->brickBlockPos;
        blockId++)
    {
        block = &segment.brickBlocks[blockId];
        connection = &segment.blockConnections[blockId];

        for(uint32_t neuronId = 0;
            neuronId < block->numberOfNeurons;
            neuronId++)
        {
            // skip section, if not active
            sourceNeuron = &block->neurons[neuronId];
            if(connection->next[neuronId].blockId == UNINIT_STATE_32) {
                continue;
            }

            // set start-values
            sourceNeuron->delta = 0.0f;
            if(sourceNeuron->active)
            {
                LocationPtr* targetLocation = &segment.blockConnections[blockId].next[neuronId];
                if(targetLocation->blockId == UNINIT_STATE_32) {
                    return;
                }

                Synapse* nextSection = segment.brickBlocks->synapseBlock.synapses[targetLocation->sectionId];
                BrickBlock* nextBlock = &segment.brickBlocks[targetLocation->blockId];
                BlockConnection* nextConnection = &segment.blockConnections[targetLocation->blockId];
                backpropagateSection(segment,
                                     nextSection,
                                     sourceNeuron,
                                     nextBlock,
                                     nextConnection,
                                     targetLocation,
                                     sourceNeuron->potential);

                sourceNeuron->delta *= 1.4427f * pow(0.5f, sourceNeuron->potential);
            }

            if(brick->isInputBrick) {
                segment.outputTransfers[sourceNeuron->targetBorderId] = sourceNeuron->delta;
            }
        }
    }
}

/**
 * @brief correct weight of synapses within a segment
 */
void
reweightCoreSegment(const CoreSegment &segment)
{
    // run back-propagation over all internal neurons and synapses
    const uint32_t numberOfBricks = segment.segmentHeader->bricks.count;
    for(int32_t pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        const uint32_t brickId = segment.brickOrder[pos];
        Brick* brick = &segment.bricks[brickId];
        if(brick->isOutputBrick)
        {
            if(backpropagateOutput(segment, brick) == false) {
                return;
            }
        }
        backpropagateNeurons(segment, brick);
    }
}

#endif // HANAMI_CORE_BACKPROPAGATION_H
