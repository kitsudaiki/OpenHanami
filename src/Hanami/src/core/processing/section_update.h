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
 * @brief searchTargetInBrick
 *
 * @param targetBrick
 *
 * @return
 */
inline SynapseConnection*
searchTargetInBrick(Brick* targetBrick)
{
    const uint64_t numberOfConnectionsBlocks = targetBrick->connectionBlocks.size();
    if (numberOfConnectionsBlocks == 0) {
        return nullptr;
    }

    uint64_t pos = rand() % numberOfConnectionsBlocks;
    for (uint64_t i = 0; i < numberOfConnectionsBlocks; i++) {
        ConnectionBlock* connectionBlock = &targetBrick->connectionBlocks[pos];

        for (uint16_t j = 0; j < NUMBER_OF_SYNAPSESECTION; j++) {
            if (connectionBlock->connections[j].origin.blockId == UNINIT_STATE_32) {
                // initialize a synapse-block if necessary
                if (connectionBlock->targetSynapseBlockPos == UNINIT_STATE_64) {
                    SynapseBlock block;
                    connectionBlock->targetSynapseBlockPos
                        = HanamiRoot::cpuSynapseBlocks.addNewItem(block);
                    if (connectionBlock->targetSynapseBlockPos == UNINIT_STATE_64) {
                        return nullptr;
                    }
                }
                return &connectionBlock->connections[j];
            }
        }
        pos = (pos + 1) % numberOfConnectionsBlocks;
    }

    return nullptr;
}

/**
 * @brief resizeConnections
 *
 * @param targetBrick
 */
inline void
resizeConnections(Brick* targetBrick)
{
    const uint32_t dimXold = targetBrick->dimX;
    const uint32_t dimYold = targetBrick->dimY;

    // update brick-dimensions
    if (targetBrick->dimX < targetBrick->numberOfNeuronBlocks) {
        targetBrick->dimX++;
    }
    targetBrick->dimY++;

    // resize list
    targetBrick->connectionBlocks.resize(targetBrick->dimX * targetBrick->dimY);

    // if there was no scaling in x-dimension, then no re-ordering necessary
    if (targetBrick->dimX == dimXold) {
        return;
    }

    LOG_DEBUG("resized connection-Block: " + std::to_string(dimXold) + ":" + std::to_string(dimYold)
              + " -> " + std::to_string(targetBrick->dimX) + ":"
              + std::to_string(targetBrick->dimY));
    uint32_t newPos = 0;
    uint32_t oldPos = 0;

    // update content of list for the new size
    for (int32_t y = dimYold - 1; y >= 1; y--) {
        for (int32_t x = dimXold - 1; x >= 0; x--) {
            newPos = (y * targetBrick->dimX) + x;
            oldPos = (y * dimXold) + x;

            targetBrick->connectionBlocks[newPos] = targetBrick->connectionBlocks[oldPos];
            targetBrick->connectionBlocks[oldPos] = ConnectionBlock();
        }
    }
}

/**
 * @brief createNewSection
 *
 * @param cluster
 * @param originLocation
 * @param offset
 * @param posInNeuron
 *
 * @return
 */
inline bool
createNewSection(Cluster& cluster, const SourceLocationPtr& originLocation, const float offset)
{
    // get origin objects
    NeuronBlock* originBlock = &cluster.neuronBlocks[originLocation.blockId];
    Neuron* originNeuron = &originBlock->neurons[originLocation.sectionId];

    const uint32_t originBrickId = cluster.neuronBlocks[originLocation.blockId].brickId;
    const Brick* originBrick = &cluster.bricks[originBrickId];
    if (originBrick->isOutputBrick) {
        return false;
    }

    // get target objects
    const uint32_t targetBrickId
        = originBrick->possibleTargetNeuronBrickIds[rand() % NUMBER_OF_POSSIBLE_NEXT];
    Brick* targetBrick = &cluster.bricks[targetBrickId];

    // get target-connection
    SynapseConnection* targetConnection = searchTargetInBrick(targetBrick);
    if (targetConnection == nullptr) {
        resizeConnections(targetBrick);
        targetConnection = searchTargetInBrick(targetBrick);
    }

    // initialize connection
    targetConnection->origin = originLocation;
    targetConnection->origin.posInNeuron = originNeuron->inUse;
    targetConnection->offset = offset;

    originNeuron->inUse++;

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
    const uint32_t numberOfBlocks = cluster.clusterHeader->neuronBlocks.count;
    for (uint32_t neuronBlockId = 0; neuronBlockId < numberOfBlocks; neuronBlockId++) {
        neuronBlock = &cluster.neuronBlocks[neuronBlockId];

        for (uint32_t sourceId = 0; sourceId < neuronBlock->numberOfNeurons; sourceId++) {
            neuron = &neuronBlock->neurons[sourceId];

            if (neuron->isNew > 0) {
                found = true;
                SourceLocationPtr originLocation;
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
