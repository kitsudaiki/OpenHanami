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
 * @brief process single update-position
 */
inline bool
processUpdatePositon_CPU(CoreSegment &segment,
                         LocationPtr* location,
                         const float offset)
{
    // get current position
    BlockConnection* currentConnection = &segment.blockConnections[location->blockId];
    const uint32_t brickId = segment.brickBlocks[location->blockId].brickId;
    Brick* currentBrick = &segment.bricks[brickId];

    // get target position
    const uint32_t randVal = rand() % 1000;
    const uint32_t targetBrickId = currentBrick->possibleTargetNeuronBrickIds[randVal];
    Brick* targetBrick = &segment.bricks[targetBrickId];
    const uint32_t targetBlockId = targetBrick->brickBlockPos + (rand() % targetBrick->numberOfNeuronSections);
    BlockConnection* targetConnection = &segment.blockConnections[targetBlockId];

    // search for available section in target-block
    uint16_t targetSectionId = 0;
    while(targetSectionId < NUMBER_OF_SYNAPSESECTION)
    {
        if(targetConnection->origin[targetSectionId].blockId == UNINIT_STATE_32) {
            break;
        }
        targetSectionId++;
    }
    if(targetSectionId == NUMBER_OF_SYNAPSESECTION) {
        return false;
    }

    // get origin position and next
    if(location->isNeuron)
    {
        targetConnection->origin[targetSectionId].blockId = location->blockId;
        targetConnection->origin[targetSectionId].sectionId = location->sectionId;
        targetConnection->offset[targetSectionId] = offset;

        currentConnection->next[location->sectionId].blockId = targetBlockId;
        currentConnection->next[location->sectionId].sectionId = targetSectionId;
    }
    else
    {
        const uint32_t originBlockId = currentConnection->origin[location->sectionId].blockId;
        const uint16_t originSectionId = currentConnection->origin[location->sectionId].blockId;

        targetConnection->origin[targetSectionId].blockId = originBlockId;
        targetConnection->origin[targetSectionId].sectionId = originSectionId;
        targetConnection->offset[targetSectionId] = offset;

        currentConnection->next[64 + location->sectionId].blockId = targetBlockId;
        currentConnection->next[64 + location->sectionId].sectionId = targetSectionId;
    }

    return true;
}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline bool
updateSections_GPU(CoreSegment &segment)
{
    /*NeuronConnection* sourceNeuronConnection = nullptr;
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

    return found;*/
    return false;
}

#endif // HANAMI_SECTION_UPDATE_H
