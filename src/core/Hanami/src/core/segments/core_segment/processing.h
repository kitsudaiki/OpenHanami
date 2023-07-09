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
createNewSynapse(SynapseSection* section,
                 Synapse* synapse,
                 const NeuronSection* targetNeuronSection,
                 const SegmentSettings* segmentSettings,
                 const float outH)
{
    const uint32_t* randomValues = HanamiRoot::m_randomValues;
    const float randMax = static_cast<float>(RAND_MAX);
    const float maxWeight = outH / static_cast<float>(segmentSettings->synapseSegmentation);
    uint32_t signRand = 0;
    const float sigNeg = segmentSettings->signNeg;

    // set activation-border
    section->connection.randomPos = (section->connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->border = maxWeight * (static_cast<float>(randomValues[section->connection.randomPos]) / randMax);

    // set target neuron
    section->connection.randomPos = (section->connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = static_cast<uint16_t>(randomValues[section->connection.randomPos]
                              % targetNeuronSection->numberOfNeurons);


    section->connection.randomPos = (section->connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = (static_cast<float>(randomValues[section->connection.randomPos]) / randMax) / 10.0f;

    // update weight with sign
    section->connection.randomPos = (section->connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[section->connection.randomPos] % 1000;
    synapse->weight *= static_cast<float>(1.0f - (1000.0f * sigNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

/**
 * @brief process synapse-section
 */
inline void
synapseProcessing(CoreSegment &segment,
                  SynapseSection* section,
                  NeuronSection* neuronSections,
                  SynapseSection* synapseSections,
                  NeuronConnection* neuronConnections,
                  SegmentSettings* segmentSettings,
                  const float outH)
{
    uint32_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    NeuronSection* targetNeuronSection = &neuronSections[section->connection.targetNeuronSectionId];
    uint8_t active = 0;
    float counter = section->connection.offset;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && outH > counter)
    {
        synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(section,
                             synapse,
                             targetNeuronSection,
                             segmentSettings,
                             outH);
        }

        // update target-neuron
        targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 126);

        // update loop-counter
        counter += synapse->border;
        pos++;
    }

    if(outH - counter > 0.01f
            && section->connection.forwardNextId == UNINIT_STATE_32)
    {
        processUpdatePositon_CPU(segment,
                                 section->connection.sourceNeuronSectionId,
                                 section->connection.sourceNeuronId,
                                 counter + section->connection.offset);
        return;
    }

    if(section->connection.forwardNextId != UNINIT_STATE_32)
    {
        synapseProcessing(segment,
                          &synapseSections[section->connection.forwardNextId],
                          neuronSections,
                          synapseSections,
                          neuronConnections,
                          segmentSettings,
                          outH);
    }
}

/**
 * @brief process only a single neuron
 */
inline void
processSingleNeuron(CoreSegment &segment,
                    const uint32_t sourceNeuronId,
                    const uint32_t sourceNeuronSectionId,
                    Neuron* neuron,
                    NeuronSection* neuronSections,
                    SynapseSection* synapseSections,
                    NeuronConnection* neuronConnections,
                    SegmentSettings* segmentSettings)
{
    // handle active-state
    if(neuron->active == 0) {
        return;
    }

    if(neuron->targetSectionId == UNINIT_STATE_32)
    {
        processUpdatePositon_CPU(segment,
                                 sourceNeuronSectionId,
                                 sourceNeuronId,
                                 0.0f);

        return;
    }

    synapseProcessing(segment,
                      &synapseSections[neuron->targetSectionId],
                      neuronSections,
                      synapseSections,
                      neuronConnections,
                      segmentSettings,
                      neuron->potential);
}

/**
 * @brief process output brick
 */
inline void
processNeuronsOfOutputBrick(const Brick* brick,
                            NeuronSection* neuronSections,
                            float* outputTransfers,
                            SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
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
            neuron = &neuronSection->neurons[neuronId];
            neuron->potential = segmentSettings->potentialOverflow * neuron->input;
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
                           const Brick* brick,
                           NeuronSection* neuronSections,
                           float* inputTransfers,
                           SynapseSection* synapseSections,
                           NeuronConnection* neuronConnections,
                           SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronSection* section = nullptr;

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
            neuron->potential = inputTransfers[neuron->targetBorderId];
            neuron->active = neuron->potential > 0.0f;

            processSingleNeuron(segment,
                                neuronId,
                                neuronSectionId,
                                neuron,
                                neuronSections,
                                synapseSections,
                                neuronConnections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process normal internal brick
 */
inline void
processNeuronsOfNormalBrick(CoreSegment &segment,
                            const Brick* brick,
                            NeuronSection* neuronSections,
                            SynapseSection* synapseSections,
                            NeuronConnection* neuronConnections,
                            SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronSection* section = nullptr;

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

            neuron->potential /= segmentSettings->neuronCooldown;
            neuron->refractionTime = neuron->refractionTime >> 1;

            if(neuron->refractionTime == 0)
            {
                neuron->potential = segmentSettings->potentialOverflow * neuron->input;
                neuron->refractionTime = segmentSettings->refractionTime;
            }

            // update neuron
            neuron->potential -= neuron->border;
            neuron->active = neuron->potential > 0.0f;
            neuron->input = 0.0f;
            neuron->potential = log2(neuron->potential + 1.0f);

            processSingleNeuron(segment,
                                neuronId,
                                neuronSectionId,
                                neuron,
                                neuronSections,
                                synapseSections,
                                neuronConnections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process all neurons within a segment
 */
inline void
prcessCoreSegment(CoreSegment &segment)
{
    Brick* bricks = segment.bricks;
    uint32_t* brickOrder = segment.brickOrder;
    NeuronSection* neuronSections = segment.neuronSections;
    SynapseSection* synapseSections = segment.synapseSections;
    NeuronConnection* neuronConnections = segment.neuronConnections;
    SegmentHeader* segmentHeader = segment.segmentHeader;
    SegmentSettings* segmentSettings = segment.segmentSettings;
    float* inputTransfers = segment.inputTransfers;
    float* outputTransfers = segment.outputTransfers;

    const uint32_t numberOfBricks = segmentHeader->bricks.count;
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        const uint32_t brickId = brickOrder[pos];
        Brick* brick = &bricks[brickId];
        if(brick->header.isInputBrick)
        {
            processNeuronsOfInputBrick(segment,
                                       brick,
                                       neuronSections,
                                       inputTransfers,
                                       synapseSections,
                                       neuronConnections,
                                       segmentSettings);
        }
        else if(brick->header.isOutputBrick)
        {
            processNeuronsOfOutputBrick(brick,
                                        neuronSections,
                                        outputTransfers,
                                        segmentSettings);
        }
        else
        {
            processNeuronsOfNormalBrick(segment,
                                        brick,
                                        neuronSections,
                                        synapseSections,
                                        neuronConnections,
                                        segmentSettings);
        }
    }
}

#endif // HANAMI_CORE_PROCESSING_H
