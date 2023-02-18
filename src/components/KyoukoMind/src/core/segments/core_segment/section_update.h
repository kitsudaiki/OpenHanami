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
 *        to be attached
 */
inline SynapseSection*
getForwardLast(const uint32_t sourceId,
               SynapseSection* sectionConnections)
{
    SynapseSection* connection = &sectionConnections[sourceId];
    if(connection->nextId == UNINIT_STATE_32) {
        return connection;
    }

    return getForwardLast(connection->nextId, sectionConnections);
}

/**
 * @brief initialize new synapse-section
 */
inline void
createNewSection(SynapseSection &result,
                 CoreSegment &segment,
                 const Brick &currentBrick)
{
    result.active = Kitsunemimi::ItemBuffer::ACTIVE_SECTION;
    result.randomPos = rand() % NUMBER_OF_RAND_VALUES;
    result.randomPos = (result.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    const uint32_t randVal = KyoukoRoot::m_randomValues[result.randomPos] % 1000;
    const uint32_t brickId = currentBrick.possibleTargetNeuronBrickIds[randVal];
    result.randomPos = (result.randomPos + 1) % NUMBER_OF_RAND_VALUES;
    result.targetNeuronSectionId = segment.bricks[brickId].neuronSectionPos;
    result.targetNeuronSectionId += KyoukoRoot::m_randomValues[result.randomPos]
                                     % segment.bricks[brickId].numberOfNeuronSections;
}

/**
 * @brief process single update-position
 */
inline void
processUpdatePositon(CoreSegment &segment,
                     const uint32_t sectionId,
                     const uint32_t neuronId)
{
    NeuronSection* sourceSection = &segment.neuronSections[sectionId];
    Brick* currentBrick = &segment.bricks[sourceSection->brickId];

    SynapseSection newSection;
    createNewSection(newSection, segment, *currentBrick);
    const uint64_t newId = segment.segmentData.addNewItem(newSection);
    if(newId == ITEM_BUFFER_UNDEFINE_POS) {
        return;
    }

    Neuron* neuron = &sourceSection->neurons[neuronId];
    if(neuron->targetSectionId == UNINIT_STATE_32) {
        neuron->targetSectionId = newId;
    } else {
        getForwardLast(neuron->targetSectionId, segment.synapseSections)->nextId = newId;
    }
}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline void
updateSections(CoreSegment &segment)
{
    UpdatePosSection* sourceUpdatePosSection = nullptr;
    UpdatePos* sourceUpdatePos = nullptr;

    // iterate over all neurons and add new synapse-section, if required
    for(uint32_t i = 0;
        i < segment.segmentHeader->updatePosSections.count;
        i++)
    {
        sourceUpdatePosSection = &segment.updatePosSections[i];
        for(uint32_t pos = 0;
            pos < sourceUpdatePosSection->numberOfPositions;
            pos++)
        {
            sourceUpdatePos = &sourceUpdatePosSection->positions[pos];
            if(sourceUpdatePos->type == 1)
            {
                sourceUpdatePos->type = 0;
                processUpdatePositon(segment, i, pos);
            }
        }
    }
}

#endif // KYOUKOMIND_SECTION_UPDATE_H
