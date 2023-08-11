/**
 * @file        processing.h
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

#ifndef HANAMI_CORE_PROCESSING_H
#define HANAMI_CORE_PROCESSING_H

#include <common.h>
#include <math.h>
#include <cmath>

#include <hanami_root.h>
#include <core/segments/brick.h>

#include "objects.h"
#include "core_segment.h"
#include "section_update.h"

/**
 * @brief initialize a new specific synapse
 */
inline void
createNewSynapse(BrickBlock* block,
                 Synapse* synapse,
                 const SegmentSettings* segmentSettings,
                 const float remainingW)
{
    const uint32_t* randomValues = HanamiRoot::m_randomValues;
    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = segmentSettings->signNeg;

    // set activation-border
    synapse->border = remainingW;

    // set target neuron
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = static_cast<uint16_t>(randomValues[block->randomPos]
                              % block->numberOfNeurons);


    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = (static_cast<float>(randomValues[block->randomPos]) / randMax) / 10.0f;

    // update weight with sign
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[block->randomPos] % 1000;
    synapse->weight *= static_cast<float>(1.0f - (1000.0f * sigNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

/**
 * @brief process synapse-section
 */
inline void
synapseProcessing(CoreSegment &segment,
                  Synapse* section,
                  BrickBlock* block,
                  BlockConnection* connection,
                  LocationPtr* sourceLocation,
                  const float outH)
{
    uint32_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    uint8_t active = 0;
    float counter = outH - connection->offset[sourceLocation->sectionId];

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && counter > 0.01f)
    {
        synapse = &section[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16) {
            createNewSynapse(block, synapse, segment.segmentSettings, counter);
        }

        if(synapse->border > 2.0f * counter
                && pos < SYNAPSES_PER_SYNAPSESECTION-2)
        {
            const float val = synapse->border / 2.0f;
            section[pos + 1].border += val;
            synapse->border -= val;
        }

        // update target-neuron
        targetNeuron = &block->neurons[synapse->targetNeuronId];
        targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 126);

        // update loop-counter
        counter -= synapse->border;
        pos++;
    }

    LocationPtr* targetLocation = &connection->next[sourceLocation->sectionId + 64];
    if(counter > 0.01f
            && targetLocation->sectionId == UNINIT_STATE_16)
    {
        const float newOffset = (outH - counter) + connection->offset[sourceLocation->sectionId];
        processUpdatePositon_CPU(segment, sourceLocation, newOffset);
    }

    if(targetLocation->sectionId != UNINIT_STATE_16)
    {
        Synapse* nextSection = block->synapseBlock.synapses[targetLocation->sectionId];
        BrickBlock* nextBlock = &segment.brickBlocks[targetLocation->blockId];
        BlockConnection* nextConnection = &segment.blockConnections[targetLocation->blockId];
        synapseProcessing(segment,  nextSection, nextBlock, nextConnection, targetLocation, outH);
    }
}

/**
 * @brief process only a single neuron
 */
inline void
processSingleNeuron(CoreSegment &segment,
                    Neuron* neuron,
                    const uint32_t blockId,
                    const uint32_t neuronId)
{
    // handle active-state
    if(neuron->active == 0) {
        return;
    }

    LocationPtr* targetLocation = &segment.blockConnections[blockId].next[neuronId];
    if(targetLocation->blockId == UNINIT_STATE_32)
    {
        LocationPtr sourceLocation;
        sourceLocation.blockId = blockId;
        sourceLocation.sectionId = neuronId;
        sourceLocation.isNeuron = true;
        if(processUpdatePositon_CPU(segment, &sourceLocation, 0.0f) == false) {
            return;
        }
    }

    Synapse* nextSection = segment.brickBlocks->synapseBlock.synapses[targetLocation->sectionId];
    BrickBlock* nextBlock = &segment.brickBlocks[targetLocation->blockId];
    BlockConnection* nextConnection = &segment.blockConnections[targetLocation->blockId];
    synapseProcessing(segment,  nextSection, nextBlock, nextConnection, targetLocation, neuron->potential);
}

/**
 * @brief process output brick
 */
inline void
processNeuronsOfOutputBrick(CoreSegment &segment,
                            const Brick* brick)
{
    Neuron* neuron = nullptr;
    BrickBlock* block = nullptr;
    float* outputTransfers = segment.outputTransfers;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronSections + brick->brickBlockPos;
        blockId++)
    {
        block = &segment.brickBlocks[blockId];
        for(uint32_t neuronId = 0;
            neuronId < block->numberOfNeurons;
            neuronId++)
        {
            neuron = &block->neurons[neuronId];
            neuron->potential = segment.segmentSettings->potentialOverflow * neuron->input;
            outputTransfers[neuron->targetBorderId] = neuron->potential;
            neuron->input = 0.0f;
        }
    }
}

/**
 * @brief process input brick
 */
inline void
processNeuronsOfInputBrick(CoreSegment &segment,
                           const Brick* brick)
{
    Neuron* neuron = nullptr;
    BrickBlock* block = nullptr;
    float* inputTransfers = segment.inputTransfers;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronSections + brick->brickBlockPos;
        blockId++)
    {
        block = &segment.brickBlocks[blockId];
        for(uint32_t neuronId = 0;
            neuronId < block->numberOfNeurons;
            neuronId++)
        {
            neuron = &block->neurons[neuronId];
            neuron->potential = inputTransfers[neuron->targetBorderId];
            neuron->active = neuron->potential > 0.0f;

            processSingleNeuron(segment, neuron, blockId, neuronId);
        }
    }
}

/**
 * @brief process normal internal brick
 */
inline void
processNeuronsOfNormalBrick(CoreSegment &segment,
                            const Brick* brick)
{
    Neuron* neuron = nullptr;
    BrickBlock* blocks = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronSections + brick->brickBlockPos;
        blockId++)
    {
        blocks = &segment.brickBlocks[blockId];
        for(uint32_t neuronId = 0;
            neuronId < blocks->numberOfNeurons;
            neuronId++)
        {
            neuron = &blocks->neurons[neuronId];

            neuron->potential /= segment.segmentSettings->neuronCooldown;
            neuron->refractionTime = neuron->refractionTime >> 1;

            if(neuron->refractionTime == 0)
            {
                neuron->potential = segment.segmentSettings->potentialOverflow * neuron->input;
                neuron->refractionTime = segment.segmentSettings->refractionTime;
            }

            // update neuron
            neuron->potential -= neuron->border;
            neuron->active = neuron->potential > 0.0f;
            neuron->input = 0.0f;
            neuron->potential = log2(neuron->potential + 1.0f);

            processSingleNeuron(segment, neuron, blockId, neuronId);
        }
    }
}

/**
 * @brief process all neurons within a segment
 */
inline void
prcessCoreSegment(CoreSegment &segment)
{
    const uint32_t numberOfBricks = segment.segmentHeader->bricks.count;
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        const uint32_t brickId = segment.brickOrder[pos];
        Brick* brick = &segment.bricks[brickId];
        if(brick->isInputBrick) {
            processNeuronsOfInputBrick(segment, brick);
        } else if(brick->isOutputBrick) {
            processNeuronsOfOutputBrick(segment, brick);
        } else {
            processNeuronsOfNormalBrick(segment, brick);
        }
    }
}

#endif // HANAMI_CORE_PROCESSING_H
