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

#ifndef KYOUKOMIND_SECTION_UPDATE_H
#define KYOUKOMIND_SECTION_UPDATE_H

#include <common.h>

#include <kyouko_root.h>
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
 * @brief search for the last entry in a chain of sections to find position where the new has
 *        to be attached in backward direction
 */
inline uint32_t
getBackwardLast(const uint32_t sourceId,
                SynapseSection* section)
{
    const uint32_t nextId = section[sourceId].connection.backwardNextId;
    if(nextId == UNINIT_STATE_32) {
        return sourceId;
    }

    return getBackwardLast(nextId, section);
}

/**
 * @brief initialize new synapse-section
 */
inline void
createNewSection(SynapseSection &result,
                 CoreSegment &segment,
                 const Brick &currentBrick,
                 const float offset)
{
    result.connection.active = Kitsunemimi::ItemBuffer::ACTIVE_SECTION;
    result.connection.randomPos = rand() % NUMBER_OF_RAND_VALUES;
    result.connection.randomPos = (result.connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    const uint32_t randVal = KyoukoRoot::m_randomValues[result.connection.randomPos] % 1000;
    const uint32_t brickId = currentBrick.possibleTargetNeuronBrickIds[randVal];
    result.connection.randomPos = (result.connection.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    result.connection.offset = offset;
    result.connection.targetNeuronSectionId = segment.bricks[brickId].neuronSectionPos;
    result.connection.targetNeuronSectionId += KyoukoRoot::m_randomValues[result.connection.randomPos]
                                               % segment.bricks[brickId].numberOfNeuronSections;
}

/**
 * @brief process single update-position
 */
inline void
processUpdatePositon(CoreSegment &segment,
                     const uint32_t sourceNeuronSectionId,
                     const uint32_t sourceNeuronId,
                     const float offset,
                     const bool useGpu)
{
    NeuronSection* sourceNeuronSection = &segment.neuronSections[sourceNeuronSectionId];
    Neuron* sourceNeuron = &sourceNeuronSection->neurons[sourceNeuronId];
    Brick* currentBrick = &segment.bricks[sourceNeuronSection->brickId];

    // create new section and connect it to buffer
    SynapseSection newSection;
    newSection.connection.sourceNeuronId = sourceNeuronId;
    newSection.connection.sourceNeuronSectionId = sourceNeuronSectionId;
    createNewSection(newSection, segment, *currentBrick, offset);
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

        if(useGpu)
        {
            std::cout<<"new id: "<<newId<<std::endl;
            segment.sectionConnections[id] = segment.synapseSections[id].connection;
            segment.sectionConnections[newId] = segment.synapseSections[newId].connection;
        }
    }

    // connect path to new section in backward-direction
    NeuronSection* targetSection = &segment.neuronSections[newSection.connection.targetNeuronSectionId];
    if(targetSection->backwardNextId == UNINIT_STATE_32)
    {
        targetSection->backwardNextId = newId;
    }
    else
    {
        const uint32_t id = getBackwardLast(targetSection->backwardNextId, segment.synapseSections);
        segment.synapseSections[id].connection.backwardNextId = newId;

        if(useGpu)
        {
            std::cout<<"new id: "<<newId<<std::endl;
            segment.sectionConnections[id] = segment.synapseSections[id].connection;
            segment.sectionConnections[newId] = segment.synapseSections[newId].connection;
        }
    }


}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline bool
updateSections(CoreSegment &segment,
               const bool useGpu)
{
    UpdatePosSection* sourceUpdatePosSection = nullptr;
    UpdatePos* sourceUpdatePos = nullptr;

    bool found = false;

    // iterate over all neurons and add new synapse-section, if required
    for(uint32_t sourceSectionId = 0;
        sourceSectionId < segment.segmentHeader->updatePosSections.count;
        sourceSectionId++)
    {
        sourceUpdatePosSection = &segment.updatePosSections[sourceSectionId];
        for(uint32_t sourceId = 0;
            sourceId < sourceUpdatePosSection->numberOfPositions;
            sourceId++)
        {
            sourceUpdatePos = &sourceUpdatePosSection->positions[sourceId];
            if(sourceUpdatePos->type == 1)
            {
                found = true;
                sourceUpdatePos->type = 0;
                processUpdatePositon(segment,
                                         sourceSectionId,
                                         sourceId,
                                         sourceUpdatePos->offset,
                                         useGpu);
            }
        }
    }

    return found;
}

#endif // KYOUKOMIND_SECTION_UPDATE_H
