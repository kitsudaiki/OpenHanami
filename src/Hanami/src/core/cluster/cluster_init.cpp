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
    initializeNeurons(cluster, clusterMeta);
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
 * @brief init all neurons with activation-border
 *
 * @return true, if successful, else false
 */
bool
initializeNeurons(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    // allocate memory for neuron-blocks
    uint32_t numberOfNeuronBlocks = 0;
    for (const BrickMeta& brickMeta : clusterMeta.bricks) {
        numberOfNeuronBlocks += calcNumberOfNeuronBlocks(brickMeta.numberOfNeurons);
    }
    cluster->neuronBlocks.resize(numberOfNeuronBlocks);
    cluster->tempNeuronBlocks.resize(numberOfNeuronBlocks);

    // initialize neuron-blocks
    for (uint32_t i = 0; i < numberOfNeuronBlocks; i++) {
        NeuronBlock newBlock;
        for (uint32_t j = 0; j < NEURONS_PER_NEURONBLOCK; j++) {
            newBlock.neurons[j].border = 0.0f;
        }
        cluster->neuronBlocks[i] = newBlock;
        cluster->tempNeuronBlocks[i] = TempNeuronBlock();
    }

    return true;
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
        InputInterface inputInterface;
        inputInterface.targetBrickId = inputMeta.targetBrickId;
        inputInterface.numberOfNeurons = inputMeta.numberOfInputs;
        inputInterface.inputNeurons = new InputNeuron[inputMeta.numberOfInputs];
        cluster->bricks[inputInterface.targetBrickId].isInputBrick = true;
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
        outputInterface.numberOfNeurons = outputMeta.numberOfOutputs;
        outputInterface.outputNeurons = new OutputNeuron[outputMeta.numberOfOutputs];
        cluster->bricks[outputInterface.targetBrickId].isOutputBrick = true;
        cluster->outputInterfaces.try_emplace(outputMeta.name, outputInterface);
    }
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
    newBrick.brickId = id;
    newBrick.brickPos = brickMeta.position;
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
initializeBricks(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta)
{
    uint32_t neuronBrickIdCounter = 0;
    uint32_t neuronBlockPosCounter = 0;
    NeuronBlock* block = nullptr;

    cluster->bricks.resize(clusterMeta.bricks.size());

    for (uint32_t i = 0; i < clusterMeta.bricks.size(); i++) {
        Brick newBrick = createNewBrick(clusterMeta.bricks.at(i), i);
        newBrick.neuronBlockPos = neuronBlockPosCounter;

        for (uint32_t j = 0; j < newBrick.numberOfNeuronBlocks; j++) {
            block = &cluster->neuronBlocks[j + neuronBlockPosCounter];
            block->brickId = newBrick.brickId;
        }

        // copy new brick to cluster
        cluster->bricks[neuronBrickIdCounter] = newBrick;
        assert(neuronBrickIdCounter == newBrick.brickId);
        neuronBrickIdCounter++;
        neuronBlockPosCounter += newBrick.numberOfNeuronBlocks;
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
