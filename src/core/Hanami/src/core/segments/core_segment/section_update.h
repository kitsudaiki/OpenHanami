/**
 * @file        section_update.h
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

#ifndef HANAMI_SECTION_UPDATE_H
#define HANAMI_SECTION_UPDATE_H

#include <common.h>

#include <hanami_root.h>
#include <core/segments/brick.h>

#include "objects.h"
#include "core_segment.h"

/**
 * @brief search for the last entry in a chain of sections to find position where the new has
 *        to be attached in forward direction
 */
inline uint32_t
getForwardLast(const uint32_t sourceId,
               SynapseSection* section)
{
    const uint32_t nextId = section[sourceId].connection.forwardNextId;
    if(nextId == UNINIT_STATE_32) {
        return sourceId;
    }

    return getForwardLast(nextId, section);
}

/**
 * @brief initialize new synapse-section
 */
inline void
createNewSection(SynapseSection &result,
                 CoreSegment &segment,
                 const uint32_t brickId,
                 const float offset)
{
    result.connection.active = Kitsunemimi::ItemBuffer::ACTIVE_SECTION;
    result.connection.randomPos = rand() % NUMBER_OF_RAND_VALUES;
    result.connection.randomPos = (result.connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    result.connection.offset = offset;
    result.connection.targetNeuronSectionId = segment.bricks[brickId].neuronSectionPos;
    result.connection.targetNeuronSectionId += HanamiRoot::m_randomValues[result.connection.randomPos]
                                               % segment.bricks[brickId].numberOfNeuronSections;
}

/**
 * @brief process single update-position
 */
inline void
processUpdatePositon_CPU(CoreSegment &segment,
                         const uint32_t sourceNeuronSectionId,
                         const uint32_t sourceNeuronId,
                         const float offset)
{
    NeuronSection* sourceNeuronSection = &segment.neuronSections[sourceNeuronSectionId];
    Neuron* sourceNeuron = &sourceNeuronSection->neurons[sourceNeuronId];
    Brick* currentBrick = &segment.bricks[sourceNeuronSection->brickId];
    const uint32_t brickId = currentBrick->getPossibleBrick();
    if(brickId == UNINIT_STATE_32) {
        return;
    }

    // create new section and connect it to buffer
    SynapseSection newSection;
    newSection.connection.sourceNeuronId = sourceNeuronId;
    newSection.connection.sourceNeuronSectionId = sourceNeuronSectionId;
    createNewSection(newSection, segment, brickId, offset);

    const uint64_t newId = segment.segmentData.addNewItem(newSection);
    if(newId == ITEM_BUFFER_UNDEFINE_POS) {
        return;
    }

    // connect path to new section in forward-direction
    if(sourceNeuron->targetSectionId == UNINIT_STATE_32)
    {
        sourceNeuron->targetSectionId = newId;
    }
    else
    {
        const uint32_t id = getForwardLast(sourceNeuron->targetSectionId, segment.synapseSections);
        segment.synapseSections[id].connection.forwardNextId = newId;
    }
}

/**
 * @brief process single update-position
 */
inline void
processUpdatePositon_GPU(CoreSegment &segment,
                         const uint32_t sourceNeuronSectionId,
                         const uint32_t sourceNeuronId,
                         const float offset)
{
    NeuronSection* sourceNeuronSection = &segment.neuronSections[sourceNeuronSectionId];
    Neuron* sourceNeuron = &sourceNeuronSection->neurons[sourceNeuronId];
    Brick* currentBrick = &segment.bricks[sourceNeuronSection->brickId];
    const uint32_t brickId = currentBrick->getPossibleBrick();
    if(brickId == UNINIT_STATE_32) {
        return;
    }

    // create new section and connect it to buffer
    SynapseSection newSection;
    newSection.connection.sourceNeuronId = sourceNeuronId;
    newSection.connection.sourceNeuronSectionId = sourceNeuronSectionId;
    createNewSection(newSection, segment, brickId, offset);
    NeuronConnection* targetCon = &segment.neuronConnections[newSection.connection.targetNeuronSectionId];
    if(targetCon->backwardIds[NEURON_CONNECTIONS-1] != UNINIT_STATE_32) {
        return;
    }

    const uint64_t newId = segment.segmentData.addNewItem(newSection);
    if(newId == ITEM_BUFFER_UNDEFINE_POS) {
        return;
    }

    segment.synapseConnections[newId] = segment.synapseSections[newId].connection;

    // connect path to new section in backward-direction
    bool found = false;
    //NeuronSection* targetSection = &segment.neuronSections[newSection.connection.targetNeuronSectionId];
    for(uint32_t i = 0; i < NEURON_CONNECTIONS; i++)
    {
        if(targetCon->backwardIds[i] == UNINIT_STATE_32)
        {
            targetCon->backwardIds[i] = newId;
            found = true;
            // std::cout<<"new-id: "<<targetSectionCon->backwardIds[i]<<std::endl;
            //std::cout<<"i: "<<(i+1)<<std::endl;
            //std::cout<<"brickID: "<<targetSection->brickId<<std::endl;
            //std::cout<<"target-section-id: "<<newSection.connection.targetNeuronSectionId<<std::endl;
            break;
        }
    }
    if(found == false)
    {
        segment.segmentData.deleteItem(newId);
        return;
    }

    // connect path to new section in forward-direction
    if(sourceNeuron->targetSectionId == UNINIT_STATE_32)
    {
        sourceNeuron->targetSectionId = newId;
    }
    else
    {
        const uint32_t id = getForwardLast(sourceNeuron->targetSectionId, segment.synapseSections);
        segment.synapseSections[id].connection.forwardNextId = newId;
        segment.synapseConnections[id] = segment.synapseSections[id].connection;
    }
}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline bool
updateSections_GPU(CoreSegment &segment)
{
    NeuronConnection* sourceNeuronConnection = nullptr;
    UpdatePos* sourceUpdatePos = nullptr;

    bool found = false;

    // iterate over all neurons and add new synapse-section, if required
    for(uint32_t sourceSectionId = 0;
        sourceSectionId < segment.segmentHeader->neuronSections.count;
        sourceSectionId++)
    {
        sourceNeuronConnection = &segment.neuronConnections[sourceSectionId];
        for(uint32_t sourceId = 0;
            sourceId < sourceNeuronConnection->numberOfPositions;
            sourceId++)
        {
            sourceUpdatePos = &sourceNeuronConnection->positions[sourceId];
            if(sourceUpdatePos->type == 1)
            {
                found = true;
                sourceUpdatePos->type = 0;
                processUpdatePositon_GPU(segment,
                                         sourceSectionId,
                                         sourceId,
                                         sourceUpdatePos->offset);
            }
        }
    }

    return found;
}

#endif // HANAMI_SECTION_UPDATE_H
