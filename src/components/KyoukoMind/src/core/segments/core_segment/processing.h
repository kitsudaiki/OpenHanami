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

#ifndef KYOUKOMIND_CORE_PROCESSING_H
#define KYOUKOMIND_CORE_PROCESSING_H

#include <common.h>
#include <math.h>
#include <cmath>

#include <kyouko_root.h>
#include <core/segments/brick.h>

#include "objects.h"
#include "core_segment.h"

/**
 * @brief initialize a new specific synapse
 */
inline void
createNewSynapse(SynapseSection* section,
                 Synapse* synapse,
                 const NeuronSection* neuronSections,
                 const SegmentSettings* segmentSettings,
                 const float remainingWeight,
                 const float outH)
{
    const uint32_t* randomValues = KyoukoRoot::m_randomValues;
    const float randMax = static_cast<float>(RAND_MAX);
    const float maxWeight = outH / static_cast<float>(segmentSettings->synapseSegmentation);
    uint32_t signRand = 0;
    const float sigNeg = segmentSettings->signNeg;

    // set activation-border
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    float newWeight = maxWeight * (static_cast<float>(randomValues[section->randomPos]) / randMax);
    synapse->border = static_cast<float>(remainingWeight < newWeight) * remainingWeight
                      + static_cast<float>(remainingWeight >= newWeight) * newWeight;

    // set target neuron
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = static_cast<uint16_t>(randomValues[section->randomPos]
                              % neuronSections[section->targetNeuronSectionId].numberOfNeurons);


    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = (static_cast<float>(randomValues[section->randomPos]) / randMax) / 10.0f;

    // update weight with sign
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[section->randomPos] % 1000;
    synapse->weight *= static_cast<float>(1.0f - (1000.0f * sigNeg > signRand) * 2);


    synapse->activeCounter = 1;
}

/**
 * @brief process synapse-section
 */
inline void
synapseProcessing(const uint32_t neuronId,
                  const uint32_t neuronSectionId,
                  SynapseSection* section,
                  const Neuron* sourceNeuron,
                  NeuronSection* neuronSections,
                  SynapseSection* synapseSections,
                  UpdatePosSection* updatePosSections,
                  SegmentSettings* segmentSettings,
                  float netH,
                  const float outH)
{
    uint32_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    uint8_t active = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && netH > 0.0f)
    {
        synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(section,
                             synapse,
                             neuronSections,
                             segmentSettings,
                             netH,
                             outH);
        }

        // update target-neuron
        targetNeuron = &(neuronSections[section->targetNeuronSectionId].neurons[synapse->targetNeuronId]);
        targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 126);

        // update loop-counter
        netH -= synapse->border;
        pos++;
    }

    if(netH > 0.01f)
    {
        if(section->nextId == UNINIT_STATE_32)
        {
            updatePosSections[neuronSectionId].positions[neuronId].type = 1;
            segmentSettings->updateSections = 1;
            return;
        }

        synapseProcessing(neuronId,
                          neuronSectionId,
                          &synapseSections[section->nextId],
                          sourceNeuron,
                          neuronSections,
                          synapseSections,
                          updatePosSections,
                          segmentSettings,
                          netH,
                          outH);
    }
}

/**
 * @brief process only a single neuron
 */
inline void
processSingleNeuron(const uint32_t neuronId,
                    const uint32_t neuronSectionId,
                    Neuron* neuron,
                    NeuronSection* neuronSections,
                    SynapseSection* synapseSections,
                    UpdatePosSection* updatePosSections,
                    SegmentSettings* segmentSettings)
{
    // handle active-state
    if(neuron->active == 0) {
        return;
    }

    if(neuron->targetSectionId == UNINIT_STATE_32)
    {
        updatePosSections[neuronSectionId].positions[neuronId].type = 1;
        segmentSettings->updateSections = 1;
        return;
    }

    synapseProcessing(neuronId,
                      neuronSectionId,
                      &synapseSections[neuron->targetSectionId],
                      neuron,
                      neuronSections,
                      synapseSections,
                      updatePosSections,
                      segmentSettings,
                      neuron->potential,
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
    NeuronSection* section = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->neuronSectionPos;
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId++)
    {
        section = &neuronSections[neuronSectionId];
        for(uint32_t neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
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
processNeuronsOfInputBrick(const Brick* brick,
                           NeuronSection* neuronSections,
                           float* inputTransfers,
                           SynapseSection* synapseSections,
                           UpdatePosSection* updatePosSections,
                           SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronSection* section = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->neuronSectionPos;
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
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

            processSingleNeuron(neuronId,
                                neuronSectionId,
                                neuron,
                                neuronSections,
                                synapseSections,
                                updatePosSections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process normal internal brick
 */
inline void
processNeuronsOfNormalBrick(const Brick* brick,
                            NeuronSection* neuronSections,
                            SynapseSection* synapseSections,
                            UpdatePosSection* updatePosSections,
                            SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronSection* section = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->neuronSectionPos;
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
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

            processSingleNeuron(neuronId,
                                neuronSectionId,
                                neuron,
                                neuronSections,
                                synapseSections,
                                updatePosSections,
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
    UpdatePosSection* updatePosSections = segment.updatePosSections;
    SegmentHeader* segmentHeader = segment.segmentHeader;
    SegmentSettings* segmentSettings = segment.segmentSettings;
    float* inputTransfers = segment.inputTransfers;
    float* outputTransfers = segment.outputTransfers;

    const uint32_t numberOfBricks = segmentHeader->bricks.count;
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        const uint32_t brickId = brickOrder[pos];
        Brick* brick = &bricks[brickId];
        if(brick->isInputBrick)
        {
            processNeuronsOfInputBrick(brick,
                                       neuronSections,
                                       inputTransfers,
                                       synapseSections,
                                       updatePosSections,
                                       segmentSettings);
        }
        else if(brick->isOutputBrick)
        {
            processNeuronsOfOutputBrick(brick,
                                        neuronSections,
                                        outputTransfers,
                                        segmentSettings);
        }
        else
        {
            processNeuronsOfNormalBrick(brick,
                                        neuronSections,
                                        synapseSections,
                                        updatePosSections,
                                        segmentSettings);
        }
    }
}

#endif // KYOUKOMIND_CORE_PROCESSING_H
