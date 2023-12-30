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
    cluster->numberOfNeuronBlocks = 0;
    Hanami::ErrorContainer error;
    uint32_t numberOfInputs = 0;
    uint32_t numberOfOutputs = 0;

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

    // create cluster metadata
    const ClusterSettings settings = initSettings(clusterMeta);
    ClusterHeader header = createNewHeader(
        clusterMeta.bricks.size(), cluster->numberOfNeuronBlocks, numberOfInputs, numberOfOutputs);

    // initialize cluster itself
    Hanami::allocateBlocks_DataBuffer(cluster->clusterData,
                                      Hanami::calcBytesToBlocks(header.staticDataSize));
    initSegmentPointer(cluster, header);
    strncpy(header.uuid.uuid, uuid.c_str(), uuid.size());

    header.settings = settings;
    cluster->clusterHeader[0] = header;

    // init content
    initializeNeurons(cluster, clusterMeta);
    addBricksToCluster(cluster, clusterMeta);
    connectAllBricks(cluster);
    initTargetBrickList(cluster);

    /*if(HanamiRoot::useOpencl)
    {
        data = new Hanami::GpuData();
        initOpencl(*data);
    }*/

    if (HanamiRoot::useCuda) {
        cluster->initCuda();
    }

    return true;
}

/**
 * @brief DynamicSegment::reinitPointer
 * @return
 */
bool
reinitPointer(Cluster* cluster, const uint64_t numberOfBytes)
{
    // TODO: checks
    uint8_t* dataPtr = static_cast<uint8_t*>(cluster->clusterData.data);

    uint64_t pos = 0;
    // uint64_t byteCounter = 0;
    cluster->clusterHeader = reinterpret_cast<ClusterHeader*>(dataPtr + pos);
    // byteCounter += sizeof(ClusterHeader);

    ClusterHeader* clusterHeader = cluster->clusterHeader;

    cluster->inputValues = new float[clusterHeader->numberOfInputs];
    cluster->outputValues = new float[clusterHeader->numberOfOutputs];
    cluster->expectedValues = new float[clusterHeader->numberOfOutputs];
    cluster->tempNeuronBlocks = new TempNeuronBlock[clusterHeader->neuronBlocks.count];

    std::fill_n(cluster->inputValues, clusterHeader->numberOfInputs, 0.0f);
    std::fill_n(cluster->outputValues, clusterHeader->numberOfOutputs, 0.0f);
    std::fill_n(cluster->expectedValues, clusterHeader->numberOfOutputs, 0.0f);
    std::fill_n(cluster->tempNeuronBlocks, clusterHeader->neuronBlocks.count, TempNeuronBlock());

    pos = clusterHeader->bricks.bytePos;
    cluster->bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    // byteCounter += clusterHeader->bricks.count * sizeof(Brick);

    cluster->namedBricks.clear();
    for (uint64_t brickId = 0; brickId < clusterHeader->bricks.count; brickId++) {
        Brick* brick = &cluster->bricks[brickId];
        cluster->namedBricks.emplace(brick->getName(), brick);
    }

    pos = clusterHeader->neuronBlocks.bytePos;
    cluster->neuronBlocks = reinterpret_cast<NeuronBlock*>(dataPtr + pos);
    // byteCounter += clusterHeader->neuronBlocks.count * sizeof(NeuronBlock);

    // check result
    // if (byteCounter != numberOfBytes - 48) {
    //    return false;
    // }

    if (HanamiRoot::useCuda) {
        cluster->initCuda();
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
ClusterSettings
initSettings(const Hanami::ClusterMeta& clusterMeta)
{
    ClusterSettings settings;

    // parse settings
    settings.synapseSegmentation = clusterMeta.synapseSegmentation;
    settings.signNeg = clusterMeta.signNeg;
    settings.maxSynapseSections = clusterMeta.maxSynapseSections;

    return settings;
}

/**
 * @brief create new cluster-header with size and position information
 *
 * @param numberOfBricks number of bricks
 * @param numberOfNeurons number of neurons
 * @param borderbufferSize size of border-buffer
 *
 * @return new cluster-header
 */
ClusterHeader
createNewHeader(const uint32_t numberOfBricks,
                const uint32_t numberOfNeuronBlocks,
                const uint64_t numberOfInputs,
                const uint64_t numberOfOutputs)
{
    ClusterHeader clusterHeader;
    uint32_t clusterDataPos = 0;

    // init header
    clusterDataPos += sizeof(ClusterHeader);

    clusterHeader.numberOfInputs = numberOfInputs;
    clusterHeader.numberOfOutputs = numberOfOutputs;

    // init bricks
    clusterHeader.bricks.count = numberOfBricks;
    clusterHeader.bricks.bytePos = clusterDataPos;
    clusterDataPos += numberOfBricks * sizeof(Brick);

    // init neuron blocks
    clusterHeader.neuronBlocks.count = numberOfNeuronBlocks;
    clusterHeader.neuronBlocks.bytePos = clusterDataPos;
    clusterDataPos += numberOfNeuronBlocks * sizeof(NeuronBlock);

    clusterHeader.staticDataSize = clusterDataPos;

    return clusterHeader;
}

/**
 * @brief init pointer within the cluster-header
 *
 * @param header cluster-header
 */
void
initSegmentPointer(Cluster* cluster, const ClusterHeader& header)
{
    uint8_t* dataPtr = static_cast<uint8_t*>(cluster->clusterData.data);
    uint64_t pos = 0;

    cluster->clusterHeader = reinterpret_cast<ClusterHeader*>(dataPtr + pos);
    cluster->clusterHeader[0] = header;

    ClusterHeader* clusterHeader = cluster->clusterHeader;

    cluster->inputValues = new float[clusterHeader->numberOfInputs];
    cluster->outputValues = new float[clusterHeader->numberOfOutputs];
    cluster->expectedValues = new float[clusterHeader->numberOfOutputs];
    cluster->tempNeuronBlocks = new TempNeuronBlock[clusterHeader->neuronBlocks.count];

    std::fill_n(cluster->inputValues, clusterHeader->numberOfInputs, 0.0f);
    std::fill_n(cluster->outputValues, clusterHeader->numberOfOutputs, 0.0f);
    std::fill_n(cluster->expectedValues, clusterHeader->numberOfOutputs, 0.0f);
    std::fill_n(cluster->tempNeuronBlocks, clusterHeader->neuronBlocks.count, TempNeuronBlock());

    pos = clusterHeader->bricks.bytePos;
    cluster->bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    std::fill_n(cluster->bricks, clusterHeader->bricks.count, Brick());

    pos = clusterHeader->neuronBlocks.bytePos;
    cluster->neuronBlocks = reinterpret_cast<NeuronBlock*>(dataPtr + pos);
    std::fill_n(cluster->neuronBlocks, clusterHeader->neuronBlocks.count, NeuronBlock());
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
connectBrick(Cluster* cluster, Brick* sourceBrick, const uint8_t side)
{
    const Hanami::Position next = getNeighborPos(sourceBrick->brickPos, side);
    // debug-output
    // std::cout<<next.x<<" : "<<next.y<<" : "<<next.z<<std::endl;

    if (next.isValid()) {
        for (uint32_t t = 0; t < cluster->clusterHeader->bricks.count; t++) {
            Brick* targetBrick = &cluster->bricks[t];
            if (targetBrick->brickPos == next) {
                sourceBrick->neighbors[side] = targetBrick->brickId;
                targetBrick->neighbors[11 - side] = sourceBrick->brickId;
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
    for (uint32_t i = 0; i < cluster->clusterHeader->bricks.count; i++) {
        Brick* sourceBrick = &cluster->bricks[i];
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
goToNextInitBrick(Cluster* cluster, Brick* currentBrick, uint32_t* maxPathLength)
{
    // check path-length to not go too far
    (*maxPathLength)--;
    if (*maxPathLength == 0) {
        return currentBrick->brickId;
    }

    // check based on the chance, if you go to the next, or not
    const float chanceForNext = 0.0f;  // TODO: make hard-coded value configurable
    if (1000.0f * chanceForNext > (rand() % 1000)) {
        return currentBrick->brickId;
    }

    // get a random possible next brick
    const uint8_t possibleNextSides[7] = {9, 3, 1, 4, 11, 5, 2};
    const uint8_t startSide = possibleNextSides[rand() % 7];
    for (uint32_t i = 0; i < 7; i++) {
        const uint8_t side = possibleNextSides[(i + startSide) % 7];
        const uint32_t nextBrickId = currentBrick->neighbors[side];
        if (nextBrickId != UNINIT_STATE_32) {
            return goToNextInitBrick(cluster, &cluster->bricks[nextBrickId], maxPathLength);
        }
    }

    // if no further next brick was found, the give back tha actual one as end of the path
    return currentBrick->brickId;
}

/**
 * @brief init target-brick-list of all bricks
 *
 * @return true, if successful, else false
 */
bool
initTargetBrickList(Cluster* cluster)
{
    for (uint32_t i = 0; i < cluster->clusterHeader->bricks.count; i++) {
        Brick* baseBrick = &cluster->bricks[i];

        // ignore output-bricks, because they only forward to the border-buffer
        // and not to other bricks
        if (baseBrick->isOutputBrick) {
            continue;
        }

        // test 1000 samples for possible next bricks
        for (uint32_t counter = 0; counter < NUMBER_OF_POSSIBLE_NEXT; counter++) {
            uint32_t maxPathLength = 2;  // TODO: make configurable
            const uint32_t brickId = goToNextInitBrick(cluster, baseBrick, &maxPathLength);
            if (brickId == baseBrick->brickId) {
                LOG_WARNING("brick has no next brick and is a dead-end. Brick-ID: "
                            + std::to_string(brickId));
            }
            baseBrick->possibleTargetNeuronBrickIds[counter] = brickId;
        }
    }

    return true;
}
