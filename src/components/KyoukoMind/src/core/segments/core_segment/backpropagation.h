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

#ifndef KYOUKOMIND_CORE_BACKPROPAGATION_H
#define KYOUKOMIND_CORE_BACKPROPAGATION_H

#include <common.h>
#include <math.h>
#include <cmath>

#include <kyouko_root.h>
#include <core/segments/brick.h>
#include <core/segments/core_segment/core_segment.h>

#include "objects.h"

/**
 * @brief backpropagate values of an output-brick
 */
inline bool
backpropagateOutput(const Brick* brick,
                    float* inputTransfers,
                    NeuronSection* neuronSections,
                    SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronSection* section = nullptr;
    float totalDelta = 0.0f;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->header.neuronSectionPos;
        neuronSectionId < brick->header.numberOfNeuronSections + brick->header.neuronSectionPos;
        neuronSectionId++)
    {
        section = &neuronSections[neuronSectionId];
        for(uint32_t neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            neuron->delta = inputTransfers[neuron->targetBorderId];
            inputTransfers[neuron->targetBorderId] = 0.0f;
            totalDelta += abs(neuron->delta);
        }
    }

    return totalDelta > segmentSettings->backpropagationBorder;
    //return true;
}

/**
 * @brief run backpropagation for a single synapse-section
 */
inline void
backpropagateSection(SynapseSection* section,
                     Neuron* sourceNeuron,
                     const float outH,
                     const Brick* brick,
                     NeuronSection* neuronSections,
                     SynapseSection* synapseSections)
{
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronSection* targetNeuronSection = &neuronSections[section->connection.targetNeuronSectionId];
    float learnValue = 0.2f;
    uint16_t pos = 0;
    float counter = section->connection.offset;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && outH > counter)
    {
        // break look, if no more synapses to process
        synapse = &section->synapses[pos];

        // update weight
        learnValue = static_cast<float>(126 - synapse->activeCounter) * 0.0002f;
        learnValue += 0.05f;
        targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        sourceNeuron->delta += targetNeuron->delta * synapse->weight;
        synapse->weight -= learnValue * targetNeuron->delta;

        counter += synapse->border;
        pos++;
    }

    if(section->connection.forwardNextId != UNINIT_STATE_32)
    {
        backpropagateSection(&synapseSections[section->connection.forwardNextId],
                             sourceNeuron,
                             outH,
                             brick,
                             neuronSections,
                             synapseSections);
    }
}

/**
 * @brief run back-propagation over all neurons
 */
inline void
backpropagateNeurons(const Brick* brick,
                     NeuronSection* neuronSections,
                     SynapseSection* synapseSections,
                     float* outputTransfers)
{
    Neuron* sourceNeuron = nullptr;
    NeuronSection* neuronSection = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->header.neuronSectionPos;
        neuronSectionId < brick->header.numberOfNeuronSections + brick->header.neuronSectionPos;
        neuronSectionId++)
    {
        neuronSection = &neuronSections[neuronSectionId];
        for(uint32_t neuronId = 0;
            neuronId < neuronSection->numberOfNeurons;
            neuronId++)
        {
            // skip section, if not active
            sourceNeuron = &neuronSection->neurons[neuronId];
            if(sourceNeuron->targetSectionId == UNINIT_STATE_32) {
                continue;
            }

            // set start-values
            sourceNeuron->delta = 0.0f;
            if(sourceNeuron->active)
            {
                backpropagateSection(&synapseSections[sourceNeuron->targetSectionId],
                                     sourceNeuron,
                                     sourceNeuron->potential,
                                     brick,
                                     neuronSections,
                                     synapseSections);

                sourceNeuron->delta *= 1.4427f * pow(0.5f, sourceNeuron->potential);
            }

            if(brick->header.isInputBrick) {
                outputTransfers[sourceNeuron->targetBorderId] = sourceNeuron->delta;
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
    Brick* bricks = segment.bricks;
    uint32_t* brickOrder = segment.brickOrder;
    NeuronSection* neuronSections = segment.neuronSections;
    SynapseSection* synapseSections = segment.synapseSections;
    SegmentHeader* segmentHeader = segment.segmentHeader;
    SegmentSettings* segmentSettings = segment.segmentSettings;
    float* inputTransfers = segment.inputTransfers;
    float* outputTransfers = segment.outputTransfers;

    // run back-propagation over all internal neurons and synapses
    const uint32_t numberOfBricks = segmentHeader->bricks.count;
    for(int32_t pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        const uint32_t brickId = brickOrder[pos];
        Brick* brick = &bricks[brickId];
        if(brick->header.isOutputBrick)
        {
            if(backpropagateOutput(brick,
                                   inputTransfers,
                                   neuronSections,
                                   segmentSettings) == false)
            {
                return;
            }
        }
        backpropagateNeurons(brick,
                             neuronSections,
                             synapseSections,
                             outputTransfers);
    }
}

#endif // KYOUKOMIND_CORE_BACKPROPAGATION_H
