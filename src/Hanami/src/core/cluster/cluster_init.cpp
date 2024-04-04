/**
 * @file        cluster_init.cpp
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

#include "cluster_init.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/processing/objects.h>
#include <core/routing_functions.h>
#include <hanami_common/logger.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

/**
 * @brief getNumberOfNeuronSections
 * @param numberOfNeurons
 * @return
 */
uint32_t
calcNumberOfNeuronBlocks(const uint32_t numberOfNeurons)
{
    uint32_t numberOfSections = numberOfNeurons / NEURONS_PER_NEURONSECTION;
    if (numberOfNeurons % NEURONS_PER_NEURONSECTION != 0) {
        numberOfSections++;
    }

    return numberOfSections;
}

/**
 * @brief initalize new cluster
 *
 * @param cluster pointer to the uninitionalized cluster
 * @param parsedContent parsed json with the information of the cluster
 * @param segmentTemplates TODO
 * @param uuid uuid for the new cluster
 *
 * @return true, if successful, else false
 */
bool
initNewCluster(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta, const std::string& uuid)
{
    Hanami::ErrorContainer error;
    uint32_t numberOfInputs = 0;
    uint32_t numberOfOutputs = 0;
    cluster->numberOfNeuronBlocks = 0;

    // calculate sizes
    uint32_t neuronsInBrick = 0;
    for (uint32_t i = 0; i < clusterMeta.bricks.size(); i++) {
        const Hanami::BrickMeta brickMeta = clusterMeta.bricks.at(i);
        neuronsInBrick = brickMeta.numberOfNeurons;
        if (brickMeta.type == Hanami::BrickType::INPUT_BRICK_TYPE) {
            numberOfInputs = brickMeta.numberOfNeurons;
        }
        if (brickMeta.type == Hanami::BrickType::OUTPUT_BRICK_TYPE) {
            numberOfOutputs = brickMeta.numberOfNeurons;
        }
        cluster->numberOfNeuronBlocks += calcNumberOfNeuronBlocks(neuronsInBrick);
    }

    // create new header
    cluster->clusterHeader = ClusterHeader();
    cluster->clusterHeader.numberOfBricks = clusterMeta.bricks.size();
    cluster->clusterHeader.numberOfNeuronBlocks = cluster->numberOfNeuronBlocks;
    cluster->clusterHeader.numberOfInputs = numberOfInputs;
    cluster->clusterHeader.numberOfOutputs = numberOfOutputs;

    // create cluster metadata
    initSettings(cluster, clusterMeta);
    initSegmentPointer(cluster);
    strncpy(cluster->clusterHeader.uuid.uuid, uuid.c_str(), uuid.size());

    // init content
    initializeNeurons(cluster, clusterMeta);
    addBricksToCluster(cluster, clusterMeta);
    connectAllBricks(cluster);
    initTargetBrickList(cluster);

    return true;
}

/**
 * @brief DynamicSegment::reinitPointer
 * @return
 */
bool
reinitPointer(Cluster* cluster, const uint64_t numberOfBytes)
{
    initSegmentPointer(cluster);

    cluster->namedBricks.clear();
    for (Brick& brick : cluster->bricks) {
        cluster->namedBricks.emplace(brick.getName(), &brick);
    }

    return true;
}

/**
 * @brief init all neurons with activation-border
 *
 * @return true, if successful, else false
 */
bool
initializeNeurons(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    uint32_t sectionPositionOffset = 0;

    for (uint32_t i = 0; i < clusterMeta.bricks.size(); i++) {
        int64_t neuronsInBrick = clusterMeta.bricks.at(i).numberOfNeurons;
        const uint32_t numberOfNeuronSectionsInBrick = calcNumberOfNeuronBlocks(neuronsInBrick);

        uint32_t sectionCounter = 0;
        while (sectionCounter < numberOfNeuronSectionsInBrick) {
            const uint32_t blockId = sectionPositionOffset + sectionCounter;
            NeuronBlock* block = &cluster->neuronBlocks[blockId];

            if (neuronsInBrick >= NEURONS_PER_NEURONSECTION) {
                for (uint32_t i = 0; i < NEURONS_PER_NEURONSECTION; i++) {
                    block->neurons[i].border = 0.0f;
                }
                block->numberOfNeurons = NEURONS_PER_NEURONSECTION;
                neuronsInBrick -= NEURONS_PER_NEURONSECTION;
            }
            else {
                for (uint32_t i = 0; i < neuronsInBrick; i++) {
                    block->neurons[i].border = 0.0f;
                }
                block->numberOfNeurons = neuronsInBrick;
                break;
            }
            sectionCounter++;
        }
        sectionPositionOffset += numberOfNeuronSectionsInBrick;
    }

    return true;
}

/**
 * @brief init sttings-block for the cluster
 *
 * @param parsedContent json-object with the cluster-description
 *
 * @return settings-object
 */
void
initSettings(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    cluster->clusterHeader.settings.neuronCooldown = clusterMeta.neuronCooldown;
    cluster->clusterHeader.settings.refractoryTime = clusterMeta.refractoryTime;
    cluster->clusterHeader.settings.maxConnectionDistance = clusterMeta.maxConnectionDistance;
    cluster->clusterHeader.settings.enableReduction = clusterMeta.enableReduction;
}

/**
 * @brief init pointer within the cluster-header
 *
 * @param header cluster-header
 */
void
initSegmentPointer(Cluster* cluster)
{
    cluster->inputValues = new float[cluster->clusterHeader.numberOfInputs];
    cluster->outputValues = new float[cluster->clusterHeader.numberOfOutputs];
    cluster->expectedValues = new float[cluster->clusterHeader.numberOfOutputs];

    std::fill_n(cluster->inputValues, cluster->clusterHeader.numberOfInputs, 0.0f);
    std::fill_n(cluster->outputValues, cluster->clusterHeader.numberOfOutputs, 0.0f);
    std::fill_n(cluster->expectedValues, cluster->clusterHeader.numberOfOutputs, 0.0f);

    for (uint32_t i = 0; i < cluster->clusterHeader.numberOfNeuronBlocks; i++) {
        cluster->tempNeuronBlocks.push_back(TempNeuronBlock());
        cluster->neuronBlocks.push_back(NeuronBlock());
    }

    cluster->bricks.resize(cluster->clusterHeader.numberOfBricks);
}

/**
 * @brief create a new brick-object
 *
 * @param brickDef json with all brick-definitions
 * @param id brick-id
 *
 * @return new brick with parsed information
 */
Brick
createNewBrick(const Hanami::BrickMeta& brickMeta, const uint32_t id)
{
    Brick newBrick;

    // copy metadata
    newBrick.brickId = id;
    newBrick.isOutputBrick = brickMeta.type == Hanami::OUTPUT_BRICK_TYPE;
    newBrick.isInputBrick = brickMeta.type == Hanami::INPUT_BRICK_TYPE;
    newBrick.setName(brickMeta.name);

    // convert other values
    newBrick.brickPos = brickMeta.position;
    newBrick.numberOfNeurons = brickMeta.numberOfNeurons;
    newBrick.numberOfNeuronBlocks = calcNumberOfNeuronBlocks(brickMeta.numberOfNeurons);

    std::fill_n(newBrick.neighbors, 12, UNINIT_STATE_32);

    return newBrick;
}

/**
 * @brief init all bricks
 *
 * @param metaBase json with all brick-definitions
 */
void
addBricksToCluster(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    uint32_t neuronBrickIdCounter = 0;
    uint32_t neuronBlockPosCounter = 0;
    NeuronBlock* block = nullptr;
    uint32_t inputBufferCounter = 0;
    uint32_t outputBufferCounter = 0;

    for (uint32_t i = 0; i < clusterMeta.bricks.size(); i++) {
        Brick newBrick = createNewBrick(clusterMeta.bricks.at(i), i);
        newBrick.neuronBlockPos = neuronBlockPosCounter;

        if (newBrick.isInputBrick) {
            newBrick.ioBufferPos = inputBufferCounter;
            inputBufferCounter += newBrick.numberOfNeurons;
        }

        if (newBrick.isOutputBrick) {
            newBrick.ioBufferPos = outputBufferCounter;
            outputBufferCounter += newBrick.numberOfNeurons;
        }

        for (uint32_t j = 0; j < newBrick.numberOfNeuronBlocks; j++) {
            block = &cluster->neuronBlocks[j + neuronBlockPosCounter];
            block->brickId = newBrick.brickId;
        }

        // copy new brick to cluster
        cluster->bricks[neuronBrickIdCounter] = newBrick;
        cluster->namedBricks.emplace(newBrick.getName(), &cluster->bricks[neuronBrickIdCounter]);
        assert(neuronBrickIdCounter == newBrick.brickId);
        neuronBrickIdCounter++;
        neuronBlockPosCounter += newBrick.numberOfNeuronBlocks;
    }

    return;
}

/**
 * @brief connect a single side of a specific brick
 *
 * @param sourceBrick pointer to the brick
 * @param side side of the brick to connect
 */
void
connectBrick(Cluster* cluster, Brick& sourceBrick, const uint8_t side)
{
    const Hanami::Position next = getNeighborPos(sourceBrick.brickPos, side);
    // debug-output
    // std::cout<<next.x<<" : "<<next.y<<" : "<<next.z<<std::endl;

    if (next.isValid()) {
        for (Brick& targetBrick : cluster->bricks) {
            if (targetBrick.brickPos == next) {
                sourceBrick.neighbors[side] = targetBrick.brickId;
                targetBrick.neighbors[11 - side] = sourceBrick.brickId;
            }
        }
    }
}

/**
 * @brief connect all breaks of the cluster
 */
void
connectAllBricks(Cluster* cluster)
{
    for (Brick& sourceBrick : cluster->bricks) {
        for (uint8_t side = 0; side < 12; side++) {
            connectBrick(cluster, sourceBrick, side);
        }
    }
}

/**
 * @brief get next possible brick
 *
 * @param currentBrick actual brick
 * @param maxPathLength maximum path length left
 *
 * @return last brick-id of the gone path
 */
uint32_t
goToNextInitBrick(Cluster* cluster, Brick& currentBrick, uint32_t* maxPathLength)
{
    // check path-length to not go too far
    (*maxPathLength)--;
    if (*maxPathLength == 0) {
        return currentBrick.brickId;
    }

    // check based on the chance, if you go to the next, or not
    const float chanceForNext = 0.0f;  // TODO: make hard-coded value configurable
    if (1000.0f * chanceForNext > (rand() % 1000)) {
        return currentBrick.brickId;
    }

    // get a random possible next brick
    const uint8_t possibleNextSides[7] = {9, 3, 1, 4, 11, 5, 2};
    const uint8_t startSide = possibleNextSides[rand() % 7];
    for (uint32_t i = 0; i < 7; i++) {
        const uint8_t side = possibleNextSides[(i + startSide) % 7];
        const uint32_t nextBrickId = currentBrick.neighbors[side];
        if (nextBrickId != UNINIT_STATE_32) {
            return goToNextInitBrick(cluster, cluster->bricks[nextBrickId], maxPathLength);
        }
    }

    // if no further next brick was found, the give back tha actual one as end of the path
    return currentBrick.brickId;
}

/**
 * @brief init target-brick-list of all bricks
 *
 * @return true, if successful, else false
 */
bool
initTargetBrickList(Cluster* cluster)
{
    for (Brick& baseBrick : cluster->bricks) {
        // ignore output-bricks, because they only forward to the border-buffer
        // and not to other bricks
        if (baseBrick.isOutputBrick) {
            continue;
        }

        // test 1000 samples for possible next bricks
        for (uint32_t counter = 0; counter < NUMBER_OF_POSSIBLE_NEXT; counter++) {
            uint32_t maxPathLength = cluster->clusterHeader.settings.maxConnectionDistance + 1;
            const uint32_t brickId = goToNextInitBrick(cluster, baseBrick, &maxPathLength);
            if (brickId == baseBrick.brickId) {
                LOG_WARNING("brick has no next brick and is a dead-end. Brick-ID: "
                            + std::to_string(brickId));
            }
            baseBrick.possibleTargetNeuronBrickIds[counter] = brickId;
        }
    }

    return true;
}
