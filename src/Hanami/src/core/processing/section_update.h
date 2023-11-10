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
#include <core/cluster/cluster.h>
#include <hanami_root.h>

#include "objects.h"

/**
 * @brief checkBackwards
 * @return
 */
inline uint32_t
checkBackwards(Cluster& cluster, const uint32_t currentId)
{
    SynapseConnection* currentConnection = &cluster.synapseConnections[currentId];
    if (currentConnection->backwardNextId == UNINIT_STATE_32) {
        return currentId;
    }

    return checkBackwards(cluster, currentConnection->backwardNextId);
}

/**
 * @brief checkForwards
 * @param cluster
 * @param currentLocation
 * @return
 */
inline LocationPtr
checkForwards(Cluster& cluster, const LocationPtr& currentLocation)
{
    SynapseConnection* currentConnection = &cluster.synapseConnections[currentLocation.blockId];
    LocationPtr nextPtr = currentConnection->next[currentLocation.sectionId];
    if (nextPtr.blockId == UNINIT_STATE_32) {
        return currentLocation;
    }

    return checkForwards(cluster, nextPtr);
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
    while (targetSectionId < NUMBER_OF_SYNAPSESECTION) {
        if (targetConnection->origin[targetSectionId].blockId == UNINIT_STATE_32) {
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
createNewSection(Cluster& cluster, const LocationPtr& originLocation, const float offset)
{
    bool sourceIsNeuron = false;
    uint32_t originBrickId = UNINIT_STATE_32;
    LocationPtr currentLocation;
    LocationPtr firstSynapseLocation
        = cluster.neuronBlocks[originLocation.blockId].neurons[originLocation.sectionId].target;

    if (firstSynapseLocation.blockId == UNINIT_STATE_32) {
        sourceIsNeuron = true;
        currentLocation = originLocation;
    }
    else {
        currentLocation = checkForwards(cluster, firstSynapseLocation);
    }
    originBrickId = cluster.neuronBlocks[originLocation.blockId].brickId;

    // get target objects
    const Brick* originBrick = &cluster.bricks[originBrickId];
    if (originBrick->isOutputBrick) {
        return false;
    }
    const uint32_t targetBrickId = originBrick->possibleTargetNeuronBrickIds[rand() % 1000];
    const Brick* targetBrick = &cluster.bricks[targetBrickId];
    const uint32_t targetNeuronBlockId
        = targetBrick->brickBlockPos + (rand() % targetBrick->numberOfNeuronBlocks);
    NeuronBlock* targetNeuronBlock = &cluster.neuronBlocks[targetNeuronBlockId];

    // get or create last available synapse-block
    uint64_t targetSynapseBlockId = UNINIT_STATE_64;
    if (targetNeuronBlock->backwardNextId == UNINIT_STATE_32) {
        SynapseConnection newConnection;
        targetSynapseBlockId = cluster.clusterData.addNewItem(newConnection);
        if (targetSynapseBlockId == ITEM_BUFFER_UNDEFINE_POS) {
            return false;
        }
        targetNeuronBlock->backwardNextId = targetSynapseBlockId;
    }
    else {
        targetSynapseBlockId = checkBackwards(cluster, targetNeuronBlock->backwardNextId);
    }

    // get possible section-id in target-block
    SynapseConnection* targetConnection = &cluster.synapseConnections[targetSynapseBlockId];
    uint32_t targetSectionId = getTargetSectionId(targetConnection);

    // in case no empty section is available, then try to create new block
    if (targetSectionId == UNINIT_STATE_32) {
        SynapseConnection newConnection;
        const uint64_t targetSynapseBlockId = cluster.clusterData.addNewItem(newConnection);
        if (targetSynapseBlockId == ITEM_BUFFER_UNDEFINE_POS) {
            return false;
        }
        targetConnection->backwardNextId = targetSynapseBlockId;
        targetConnection = &cluster.synapseConnections[targetSynapseBlockId];
        targetSectionId = 0;
    }

    // update connection
    if (sourceIsNeuron) {
        NeuronBlock* neuronBlock = &cluster.neuronBlocks[currentLocation.blockId];
        neuronBlock->neurons[currentLocation.sectionId].target.blockId = targetSynapseBlockId;
        neuronBlock->neurons[currentLocation.sectionId].target.sectionId = targetSectionId;
    }
    else {
        SynapseConnection* currentConnection = &cluster.synapseConnections[currentLocation.blockId];
        currentConnection->next[currentLocation.sectionId].blockId = targetSynapseBlockId;
        currentConnection->next[currentLocation.sectionId].sectionId = targetSectionId;
    }

    targetConnection->origin[targetSectionId].blockId = originLocation.blockId;
    targetConnection->origin[targetSectionId].sectionId = originLocation.sectionId;
    targetConnection->offset[targetSectionId] = offset;
    targetConnection->targetNeuronBlockId = targetNeuronBlockId;

    return true;
}

/**
 * @brief prcess update-positions in order to create new sections
 */
inline bool
updateSections(Cluster& cluster)
{
    NeuronBlock* neuronBlock = nullptr;
    Neuron* neuron = nullptr;

    bool found = false;

    // iterate over all neurons and add new synapse-section, if required
    for (uint32_t neuronBlockId = 0; neuronBlockId < cluster.clusterHeader->neuronBlocks.count;
         neuronBlockId++)
    {
        neuronBlock = &cluster.neuronBlocks[neuronBlockId];
        for (uint32_t sourceId = 0; sourceId < neuronBlock->numberOfNeurons; sourceId++) {
            neuron = &neuronBlock->neurons[sourceId];
            if (neuron->isNew) {
                found = true;
                LocationPtr originLocation;
                originLocation.blockId = neuronBlockId;
                originLocation.sectionId = sourceId;

                createNewSection(cluster, originLocation, neuron->newOffset);

                neuron->newOffset = 0.0f;
                neuron->isNew = 0;
            }
        }
    }

    return found;
}

#endif  // HANAMI_SECTION_UPDATE_H
