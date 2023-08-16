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
#include "objects.h"

#include <core/cluster/cluster.h>

/**
 * @brief checkBackwards
 * @return
 */
inline uint32_t
checkBackwards(Cluster &cluster, const uint32_t nextId)
{
    SynapseConnection* nextConnection = &cluster.synapseConnections[nextId];
    if(nextConnection->backwardNextId == UNINIT_STATE_32) {
        return nextId;
    }

    return checkBackwards(cluster, nextConnection->backwardNextId);
}

/**
 * @brief getTargetSectionId
 * @param targetConnection
 * @return
 */
inline uint32_t
getTargetSectionId(SynapseConnection* targetConnection)
{
    uint16_t targetSectionId = 0;
    while(targetSectionId < NUMBER_OF_SYNAPSESECTION)
    {
        if(targetConnection->origin[targetSectionId].blockId == UNINIT_STATE_32) {
            return targetSectionId;
        }
        targetSectionId++;
    }

    return UNINIT_STATE_32;
}

/**
 * @brief process single update-position
 */
inline bool
processUpdatePositon_CPU(Cluster &cluster,
                         const LocationPtr* currentLocation,
                         LocationPtr* targetLocation,
                         const float offset)
{
    // get current position
    uint32_t originBrickId = UNINIT_STATE_32;

    if(currentLocation->isNeuron)
    {
        NeuronBlock* neuronBlock = &cluster.neuronBlocks[currentLocation->blockId];
        originBrickId = neuronBlock->brickId;
    }
    else
    {
        SynapseConnection* currentConnection = &cluster.synapseConnections[currentLocation->blockId];
        const uint32_t originBlockId = currentConnection->origin[currentLocation->sectionId].blockId;
        originBrickId = cluster.neuronBlocks[originBlockId].brickId;
    }

    // get target objects
    const Brick* originBrick = &cluster.bricks[originBrickId];
    const uint32_t targetBrickId = originBrick->possibleTargetNeuronBrickIds[rand() % 1000];
    const Brick* targetBrick = &cluster.bricks[targetBrickId];
    const uint32_t targetNeuronBlockId = targetBrick->brickBlockPos + (rand() % targetBrick->numberOfNeuronSections);
    NeuronBlock* targetNeuronBlock = &cluster.neuronBlocks[targetNeuronBlockId];

    // get or create last available synapse-block
    uint64_t targetSynapseBlockId = UNINIT_STATE_64;
    if(targetNeuronBlock->backwardNextId == UNINIT_STATE_32)
    {
        SynapseConnection newConnection;
        targetSynapseBlockId = cluster.clusterData.addNewItem(newConnection);
        if(targetSynapseBlockId == ITEM_BUFFER_UNDEFINE_POS) {
            return false;
        }
    }
    else
    {
        targetSynapseBlockId = checkBackwards(cluster, targetNeuronBlock->backwardNextId);
    }

    // get possible section-id in target-block
    SynapseConnection* targetConnection = &cluster.synapseConnections[targetSynapseBlockId];
    const uint32_t targetSectionId = getTargetSectionId(targetConnection);

    // in case no empty section is available, then try to create new block
    if(targetSectionId == NUMBER_OF_SYNAPSESECTION)
    {
        SynapseConnection newConnection;
        const uint64_t targetSynapseBlockId = cluster.clusterData.addNewItem(newConnection);
        if(targetSynapseBlockId == ITEM_BUFFER_UNDEFINE_POS) {
            return false;
        }
        targetConnection = &cluster.synapseConnections[targetSynapseBlockId];
    }

    targetLocation->blockId = targetSynapseBlockId;
    targetLocation->sectionId = targetSectionId;

    // update connection
    if(currentLocation->isNeuron)
    {
        NeuronBlock* neuronBlock = &cluster.neuronBlocks[currentLocation->blockId];

        targetConnection->origin[targetSectionId].blockId = currentLocation->blockId;
        targetConnection->origin[targetSectionId].sectionId = currentLocation->sectionId;
        targetConnection->offset[targetSectionId] = offset;
        targetConnection->targetNeuronBlockId = targetNeuronBlockId;

        neuronBlock->neurons[currentLocation->sectionId].target.blockId = targetSynapseBlockId;
        neuronBlock->neurons[currentLocation->sectionId].target.sectionId = targetSectionId;
    }
    else
    {
        SynapseConnection* currentConnection = &cluster.synapseConnections[currentLocation->blockId];
        const uint32_t originBlockId = currentConnection->origin[currentLocation->sectionId].blockId;
        const uint16_t originSectionId = currentConnection->origin[currentLocation->sectionId].blockId;

        targetConnection->origin[targetSectionId].blockId = originBlockId;
        targetConnection->origin[targetSectionId].sectionId = originSectionId;
        targetConnection->offset[targetSectionId] = offset;
        targetConnection->targetNeuronBlockId = targetNeuronBlockId;

        currentConnection->next[currentLocation->sectionId].blockId = targetSynapseBlockId;
        currentConnection->next[currentLocation->sectionId].sectionId = targetSectionId;
    }

    return true;
}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline bool
updateSections_GPU(Cluster &cluster)
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
