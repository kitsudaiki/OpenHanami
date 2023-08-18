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

#include <core/processing/objects.h>
#include <core/processing/objects.h>

#include <core/routing_functions.h>
#include <core/cluster/cluster_init.h>

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiJson/json_item.h>

/**
 * @brief getNumberOfNeuronSections
 * @param numberOfNeurons
 * @return
 */
uint32_t
getNumberOfNeuronSections(const uint32_t numberOfNeurons)
{
    uint32_t numberOfSections = numberOfNeurons / NEURONS_PER_NEURONSECTION;
    if(numberOfNeurons % NEURONS_PER_NEURONSECTION != 0) {
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
initNewCluster(Cluster* cluster,
               const Kitsunemimi::Hanami::ClusterMeta &clusterMeta,
               const std::string &uuid)
{
    cluster->numberOfBrickBlocks = 0;
    Kitsunemimi::ErrorContainer error;
    uint32_t numberOfInputs = 0;
    uint32_t numberOfOutputs = 0;

    // calculate sizes
    uint32_t neuronsInBrick = 0;
    for(uint32_t i = 0; i < clusterMeta.bricks.size(); i++)
    {
        const Kitsunemimi::Hanami::BrickMeta brickMeta = clusterMeta.bricks.at(i);
        neuronsInBrick = brickMeta.numberOfNeurons;
        if(brickMeta.type == Kitsunemimi::Hanami::BrickType::INPUT_BRICK_TYPE) {
            numberOfInputs = brickMeta.numberOfNeurons;;
        }
        if(brickMeta.type == Kitsunemimi::Hanami::BrickType::OUTPUT_BRICK_TYPE) {
            numberOfOutputs = brickMeta.numberOfNeurons;;
        }
        cluster->numberOfBrickBlocks += getNumberOfNeuronSections(neuronsInBrick);
    }

    // create segment metadata
    const SegmentSettings settings = initSettings(clusterMeta);
    ClusterHeader header = createNewHeader(clusterMeta.bricks.size(),
                                           cluster->numberOfBrickBlocks,
                                           settings.maxSynapseSections,
                                           numberOfInputs,
                                           numberOfOutputs);

    // initialize segment itself
    allocateSegment(cluster, header);
    initSegmentPointer(cluster, header);
    strncpy(header.uuid.uuid, uuid.c_str(), uuid.size());

    cluster->clusterSettings[0] = settings;
    cluster->clusterHeader[0] = header;

    // init content
    initializeNeurons(cluster, clusterMeta);
    addBricksToSegment(cluster, clusterMeta);
    connectAllBricks(cluster);
    initTargetBrickList(cluster);

    /*if(HanamiRoot::useOpencl)
    {
        data = new Kitsunemimi::GpuData();
        initOpencl(*data);
    }

    if(HanamiRoot::useCuda) {
        initCuda();
    }*/

    return true;
}

/**
 * @brief DynamicSegment::reinitPointer
 * @return
 */
bool
reinitPointer(Cluster* cluster,
              const uint64_t numberOfBytes)
{
    // TODO: checks
    uint8_t* dataPtr = static_cast<uint8_t*>(cluster->clusterData.staticData);

    uint64_t pos = 0;
    uint64_t byteCounter = 0;
    cluster->clusterHeader = reinterpret_cast<ClusterHeader*>(dataPtr + pos);
    byteCounter += sizeof(ClusterHeader);

    ClusterHeader* clusterHeader = cluster->clusterHeader;

    pos = clusterHeader->settings.bytePos;
    cluster->clusterSettings = reinterpret_cast<SegmentSettings*>(dataPtr + pos);
    byteCounter += sizeof(SegmentSettings);

    pos = clusterHeader->inputValues.bytePos;
    cluster->inputValues = reinterpret_cast<float*>(dataPtr + pos);
    byteCounter += clusterHeader->inputValues.count * sizeof(float);

    pos = clusterHeader->outputValues.bytePos;
    cluster->outputValues = reinterpret_cast<float*>(dataPtr + pos);
    byteCounter += clusterHeader->outputValues.count * sizeof(float);

    pos = clusterHeader->expectedValues.bytePos;
    cluster->expectedValues = reinterpret_cast<float*>(dataPtr + pos);
    byteCounter += clusterHeader->expectedValues.count * sizeof(float);

    pos = clusterHeader->bricks.bytePos;
    cluster->bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    byteCounter += clusterHeader->bricks.count * sizeof(Brick);

    pos = clusterHeader->brickOrder.bytePos;
    cluster->brickOrder = reinterpret_cast<uint32_t*>(dataPtr + pos);
    byteCounter += clusterHeader->brickOrder.count * sizeof(uint32_t);

    pos = clusterHeader->neuronBlocks.bytePos;
    cluster->neuronBlocks = reinterpret_cast<NeuronBlock*>(dataPtr + pos);
    byteCounter += clusterHeader->neuronBlocks.count * sizeof(NeuronBlock);

    pos = clusterHeader->synapseBlocks.bytePos;
    cluster->synapseBlocks = reinterpret_cast<SynapseBlock*>(dataPtr + pos);
    byteCounter += clusterHeader->synapseBlocks.count * sizeof(SynapseBlock);

    dataPtr = static_cast<uint8_t*>(cluster->clusterData.itemData);
    //pos = segmentHeader->synapseSections.bytePos;
    cluster->synapseConnections = reinterpret_cast<SynapseConnection*>(dataPtr);
    byteCounter += clusterHeader->synapseConnections.count * sizeof(SynapseConnection);

    /*if(HanamiRoot::useOpencl)
    {
        data = new Kitsunemimi::GpuData();
        initOpencl(*data);
    }

    if(HanamiRoot::useCuda) {
        initCuda();
    }*/

    // check result
    if(byteCounter != numberOfBytes - 48) {
        return false;
    }

    return true;
}

/**
 * @brief init all neurons with activation-border
 *
 * @return true, if successful, else false
 */
bool
initializeNeurons(Cluster* cluster,
                  const Kitsunemimi::Hanami::ClusterMeta &clusterMeta)
{
    uint32_t sectionPositionOffset = 0;

    for(uint32_t i = 0; i < clusterMeta.bricks.size(); i++)
    {
        int64_t neuronsInBrick = clusterMeta.bricks.at(i).numberOfNeurons;
        const uint32_t numberOfNeuronSectionsInBrick = getNumberOfNeuronSections(neuronsInBrick);

        uint32_t sectionCounter = 0;
        while(sectionCounter < numberOfNeuronSectionsInBrick)
        {
            const uint32_t blockId = sectionPositionOffset + sectionCounter;
            NeuronBlock* block = &cluster->neuronBlocks[blockId];

            if(neuronsInBrick >= NEURONS_PER_NEURONSECTION)
            {
                for(uint32_t i = 0; i < NEURONS_PER_NEURONSECTION; i++) {
                    block->neurons[i].border = 0.0f;
                }
                block->numberOfNeurons = NEURONS_PER_NEURONSECTION;
                neuronsInBrick -= NEURONS_PER_NEURONSECTION;
            }
            else
            {
                for(uint32_t i = 0; i < neuronsInBrick; i++) {
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
 * @brief init sttings-block for the segment
 *
 * @param parsedContent json-object with the segment-description
 *
 * @return settings-object
 */
SegmentSettings
initSettings(const Kitsunemimi::Hanami::ClusterMeta &clusterMeta)
{
    SegmentSettings settings;

    // parse settings
    settings.synapseSegmentation = clusterMeta.synapseSegmentation;
    settings.signNeg = clusterMeta.signNeg;
    settings.maxSynapseSections = clusterMeta.maxSynapseSections;

    return settings;
}

/**
 * @brief create new segment-header with size and position information
 *
 * @param numberOfBricks number of bricks
 * @param numberOfNeurons number of neurons
 * @param borderbufferSize size of border-buffer
 *
 * @return new segment-header
 */
ClusterHeader
createNewHeader(const uint32_t numberOfBricks,
                const uint32_t numberOfBrickBlocks,
                const uint32_t maxSynapseSections,
                const uint64_t numberOfInputs,
                const uint64_t numberOfOutputs)
{
    ClusterHeader clusterHeader;
    uint32_t clusterDataPos = 0;

    // init header
    clusterDataPos += sizeof(ClusterHeader);

    // init settings
    clusterHeader.settings.count = 1;
    clusterHeader.settings.bytePos = clusterDataPos;
    clusterDataPos += sizeof(SegmentSettings);

    // init inputTransfers
    clusterHeader.inputValues.count = numberOfInputs;
    clusterHeader.inputValues.bytePos = clusterDataPos;
    clusterDataPos += numberOfInputs * sizeof(float);

    // init numberOfOutputs
    clusterHeader.outputValues.count = numberOfOutputs;
    clusterHeader.outputValues.bytePos = clusterDataPos;
    clusterDataPos += numberOfOutputs * sizeof(float);

    // init numberOfExprextes
    clusterHeader.expectedValues.count = numberOfOutputs;
    clusterHeader.expectedValues.bytePos = clusterDataPos;
    clusterDataPos += numberOfOutputs * sizeof(float);

    // init bricks
    clusterHeader.bricks.count = numberOfBricks;
    clusterHeader.bricks.bytePos = clusterDataPos;
    clusterDataPos += numberOfBricks * sizeof(Brick);

    // init brick-order
    clusterHeader.brickOrder.count = numberOfBricks;
    clusterHeader.brickOrder.bytePos = clusterDataPos;
    clusterDataPos += numberOfBricks * sizeof(uint32_t);

    // init neuron blocks
    clusterHeader.neuronBlocks.count = numberOfBrickBlocks;
    clusterHeader.neuronBlocks.bytePos = clusterDataPos;
    clusterDataPos += numberOfBrickBlocks * sizeof(NeuronBlock);

    // init synapse blocks
    clusterHeader.synapseBlocks.count = maxSynapseSections;
    clusterHeader.synapseBlocks.bytePos = clusterDataPos;
    clusterDataPos += maxSynapseSections * sizeof(SynapseBlock);

    clusterHeader.staticDataSize = clusterDataPos;

    // init synapse sections
    clusterDataPos = 0;
    clusterHeader.synapseConnections.count = maxSynapseSections;
    clusterHeader.synapseConnections.bytePos = clusterDataPos;

    return clusterHeader;
}

/**
 * @brief init pointer within the segment-header
 *
 * @param header segment-header
 */
void
initSegmentPointer(Cluster* cluster,
                   const ClusterHeader &header)
{
    uint8_t* dataPtr = static_cast<uint8_t*>(cluster->clusterData.staticData);
    uint64_t pos = 0;

    cluster->clusterHeader = reinterpret_cast<ClusterHeader*>(dataPtr + pos);
    cluster->clusterHeader[0] = header;

    ClusterHeader* clusterHeader = cluster->clusterHeader;

    pos = clusterHeader->settings.bytePos;
    cluster->clusterSettings = reinterpret_cast<SegmentSettings*>(dataPtr + pos);

    pos = clusterHeader->inputValues.bytePos;
    cluster->inputValues = reinterpret_cast<float*>(dataPtr + pos);
    std::fill_n(cluster->inputValues, clusterHeader->inputValues.count, 0.0f);

    pos = clusterHeader->outputValues.bytePos;
    cluster->outputValues = reinterpret_cast<float*>(dataPtr + pos);
    std::fill_n(cluster->outputValues, clusterHeader->outputValues.count, 0.0f);

    pos = clusterHeader->expectedValues.bytePos;
    cluster->expectedValues = reinterpret_cast<float*>(dataPtr + pos);
    std::fill_n(cluster->expectedValues, clusterHeader->expectedValues.count, 0.0f);

    pos = clusterHeader->bricks.bytePos;
    cluster->bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    std::fill_n(cluster->bricks, clusterHeader->bricks.count, Brick());

    pos = clusterHeader->brickOrder.bytePos;
    cluster->brickOrder = reinterpret_cast<uint32_t*>(dataPtr + pos);
    for(uint32_t i = 0; i < clusterHeader->bricks.count; i++) {
        cluster->brickOrder[i] = i;
    }

    pos = clusterHeader->neuronBlocks.bytePos;
    cluster->neuronBlocks = reinterpret_cast<NeuronBlock*>(dataPtr + pos);
    std::fill_n(cluster->neuronBlocks, clusterHeader->neuronBlocks.count, NeuronBlock());

    pos = clusterHeader->synapseBlocks.bytePos;
    cluster->synapseBlocks = reinterpret_cast<SynapseBlock*>(dataPtr + pos);
    std::fill_n(cluster->synapseBlocks, clusterHeader->synapseBlocks.count, SynapseBlock());

    dataPtr = static_cast<uint8_t*>(cluster->clusterData.itemData);
    pos = clusterHeader->synapseConnections.bytePos;
    cluster->synapseConnections = reinterpret_cast<SynapseConnection*>(dataPtr + pos);
}

/**
 * @brief allocate memory for the segment
 *
 * @param header header with the size-information
 */
void
allocateSegment(Cluster* cluster,
                ClusterHeader &header)
{
    cluster->clusterData.initBuffer<SynapseConnection>(header.synapseConnections.count, header.staticDataSize);
    cluster->clusterData.deleteAll();
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
createNewBrick(const Kitsunemimi::Hanami::BrickMeta &brickMeta,
               const uint32_t id)
{
    Brick newBrick;

    // copy metadata
    newBrick.brickId = id;
    newBrick.isOutputBrick = brickMeta.type == Kitsunemimi::Hanami::OUTPUT_BRICK_TYPE;
    newBrick.isInputBrick = brickMeta.type == Kitsunemimi::Hanami::INPUT_BRICK_TYPE;

    // convert other values
    newBrick.brickPos = brickMeta.position;
    newBrick.numberOfNeurons = brickMeta.numberOfNeurons;
    newBrick.numberOfNeuronBlocks = getNumberOfNeuronSections(brickMeta.numberOfNeurons);

    std::fill_n(newBrick.neighbors, 12, UNINIT_STATE_32);

    return newBrick;
}

/**
 * @brief init all bricks
 *
 * @param metaBase json with all brick-definitions
 */
void
addBricksToSegment(Cluster* cluster,
                   const Kitsunemimi::Hanami::ClusterMeta &clusterMeta)
{
    uint32_t neuronBrickIdCounter = 0;
    uint32_t neuronSectionPosCounter = 0;
    NeuronBlock* block = nullptr;
    uint32_t neuronIdCounter = 0;

    for(uint32_t i = 0; i < clusterMeta.bricks.size(); i++)
    {
        Brick newBrick = createNewBrick(clusterMeta.bricks.at(i), i);
        newBrick.brickBlockPos = neuronSectionPosCounter;

        for(uint32_t j = 0; j < newBrick.numberOfNeuronBlocks; j++)
        {
            block = &cluster->neuronBlocks[j + neuronSectionPosCounter];
            block->brickId = newBrick.brickId;
            for(uint32_t k = 0; k < block->numberOfNeurons; k++) {
                neuronIdCounter++;
            }
        }

        // copy new brick to segment
        cluster->bricks[neuronBrickIdCounter] = newBrick;
        assert(neuronBrickIdCounter == newBrick.brickId);
        neuronBrickIdCounter++;
        neuronSectionPosCounter += newBrick.numberOfNeuronBlocks;
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
connectBrick(Cluster* cluster,
             Brick* sourceBrick,
             const uint8_t side)
{
    const Kitsunemimi::Position next = getNeighborPos(sourceBrick->brickPos, side);
    // debug-output
    // std::cout<<next.x<<" : "<<next.y<<" : "<<next.z<<std::endl;

    if(next.isValid())
    {
        for(uint32_t t = 0; t < cluster->clusterHeader->bricks.count; t++)
        {
            Brick* targetBrick = &cluster->bricks[t];
            if(targetBrick->brickPos == next)
            {
                sourceBrick->neighbors[side] = targetBrick->brickId;
                targetBrick->neighbors[11 - side] = sourceBrick->brickId;
            }
        }
    }
}

/**
 * @brief connect all breaks of the segment
 */
void
connectAllBricks(Cluster* cluster)
{
    for(uint32_t i = 0; i < cluster->clusterHeader->bricks.count; i++)
    {
        Brick* sourceBrick = &cluster->bricks[i];
        for(uint8_t side = 0; side < 12; side++) {
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
goToNextInitBrick(Cluster* cluster,
                  Brick* currentBrick,
                  uint32_t* maxPathLength)
{
    // check path-length to not go too far
    (*maxPathLength)--;
    if(*maxPathLength == 0) {
        return currentBrick->brickId;
    }

    // check based on the chance, if you go to the next, or not
    const float chanceForNext = 0.0f;  // TODO: make hard-coded value configurable
    if(1000.0f * chanceForNext > (rand() % 1000)) {
        return currentBrick->brickId;
    }

    // get a random possible next brick
    const uint8_t possibleNextSides[7] = {9, 3, 1, 4, 11, 5, 2};
    const uint8_t startSide = possibleNextSides[rand() % 7];
    for(uint32_t i = 0; i < 7; i++)
    {
        const uint8_t side = possibleNextSides[(i + startSide) % 7];
        const uint32_t nextBrickId = currentBrick->neighbors[side];
        if(nextBrickId != UNINIT_STATE_32) {
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
    for(uint32_t i = 0; i < cluster->clusterHeader->bricks.count; i++)
    {
        Brick* baseBrick = &cluster->bricks[i];

        // ignore output-bricks, because they only forward to the border-buffer
        // and not to other bricks
        if(baseBrick->isOutputBrick) {
            continue;
        }

        // test 1000 samples for possible next bricks
        for(uint32_t counter = 0; counter < 1000; counter++)
        {
            uint32_t maxPathLength = 2; // TODO: make configurable
            const uint32_t brickId = goToNextInitBrick(cluster, baseBrick, &maxPathLength);
            if(brickId == baseBrick->brickId)
            {
                LOG_WARNING("brick has no next brick and is a dead-end. Brick-ID: "
                            + std::to_string(brickId));
            }
            baseBrick->possibleTargetNeuronBrickIds[counter] = brickId;
        }
    }

    return true;
}
