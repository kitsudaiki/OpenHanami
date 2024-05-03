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
    uint32_t numberOfSections = numberOfNeurons / NEURONS_PER_NEURONBLOCK;
    if (numberOfNeurons % NEURONS_PER_NEURONBLOCK != 0) {
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
    initializeHeader(cluster, uuid);
    initializeSettings(cluster, clusterMeta);
    initializeBricks(cluster, clusterMeta);
    initializeInputs(cluster, clusterMeta);
    initializeOutputs(cluster, clusterMeta);

    return true;
}

/**
 * @brief DynamicSegment::reinitPointer
 * @return
 */
bool
reinitPointer(Cluster* cluster, const uint64_t numberOfBytes)
{
    return true;
}

/**
 * @brief initializeHeader
 * @param cluster
 * @param uuid
 */
void
initializeHeader(Cluster* cluster, const std::string& uuid)
{
    cluster->clusterHeader = ClusterHeader();
    strncpy(cluster->clusterHeader.uuid.uuid, uuid.c_str(), uuid.size());
}

/**
 * @brief init sttings-block for the cluster
 *
 * @param parsedContent json-object with the cluster-description
 *
 * @return settings-object
 */
void
initializeSettings(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    cluster->clusterHeader.settings.neuronCooldown = clusterMeta.neuronCooldown;
    cluster->clusterHeader.settings.refractoryTime = clusterMeta.refractoryTime;
    cluster->clusterHeader.settings.maxConnectionDistance = clusterMeta.maxConnectionDistance;
    cluster->clusterHeader.settings.enableReduction = clusterMeta.enableReduction;
}

/**
 * @brief initializeInputs
 * @param cluster
 * @param clusterMeta
 */
void
initializeInputs(Cluster* cluster, const ClusterMeta& clusterMeta)
{
    for (const InputMeta& inputMeta : clusterMeta.inputs) {
        uint32_t numberOfNeuronBlocks = (inputMeta.numberOfInputs / NEURONS_PER_NEURONBLOCK) + 1;
        InputInterface inputInterface;
        inputInterface.targetBrickId = inputMeta.targetBrickId;
        inputInterface.numberOfInputNeurons = inputMeta.numberOfInputs;
        inputInterface.inputNeurons = new InputNeuron[inputMeta.numberOfInputs];
        cluster->bricks[inputInterface.targetBrickId].isInputBrick = true;
        cluster->bricks[inputInterface.targetBrickId].neuronBlocks.resize(numberOfNeuronBlocks);
        cluster->bricks[inputInterface.targetBrickId].tempNeuronBlocks.resize(numberOfNeuronBlocks);
        cluster->inputInterfaces.try_emplace(inputMeta.name, inputInterface);
    }
}

/**
 * @brief initializeOutputs
 * @param cluster
 * @param clusterMeta
 */
void
initializeOutputs(Cluster* cluster, const ClusterMeta& clusterMeta)
{
    for (const OutputMeta& outputMeta : clusterMeta.outputs) {
        OutputInterface outputInterface;
        outputInterface.targetBrickId = outputMeta.targetBrickId;
        outputInterface.numberOfOutputNeurons = outputMeta.numberOfOutputs;
        outputInterface.outputNeurons = new OutputNeuron[outputMeta.numberOfOutputs];
        cluster->bricks[outputInterface.targetBrickId].isOutputBrick = true;
        cluster->outputInterfaces.try_emplace(outputMeta.name, outputInterface);
    }
}

/**
 * @brief init all bricks
 *
 * @param metaBase json with all brick-definitions
 */
void
initializeBricks(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    cluster->bricks.resize(clusterMeta.bricks.size());

    for (uint32_t i = 0; i < clusterMeta.bricks.size(); i++) {
        Brick newBrick;
        newBrick.brickId = i;
        newBrick.brickPos = clusterMeta.bricks.at(i).position;
        std::fill_n(newBrick.neighbors, 12, UNINIT_STATE_32);

        cluster->bricks[i] = newBrick;
    }

    connectAllBricks(cluster);
    initializeTargetBrickList(cluster);

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
goToNextInitBrick(Cluster* cluster, Brick& currentBrick, uint32_t& maxPathLength)
{
    // check path-length to not go too far
    maxPathLength--;
    if (maxPathLength == 0) {
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
 */
void
initializeTargetBrickList(Cluster* cluster)
{
    for (Brick& baseBrick : cluster->bricks) {
        for (uint32_t counter = 0; counter < NUMBER_OF_POSSIBLE_NEXT; counter++) {
            uint32_t maxPathLength = cluster->clusterHeader.settings.maxConnectionDistance + 1;
            const uint32_t brickId = goToNextInitBrick(cluster, baseBrick, maxPathLength);
            if (baseBrick.brickId != brickId) {
                baseBrick.possibleBrickTargetIds[counter] = brickId;
            }
            else {
                baseBrick.possibleBrickTargetIds[counter] = UNINIT_STATE_32;
            }
        }
    }
}
