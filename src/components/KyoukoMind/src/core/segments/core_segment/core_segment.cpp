/**
 * @file        core_segment.cpp
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

#include "core_segment.h"

#include <core/routing_functions.h>
#include <gpu_kernel.h>

#include <core/segments/core_segment/section_update.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiOpencl/gpu_interface.h>
#include <libKitsunemimiOpencl/gpu_handler.h>

#include <libKitsunemimiHanamiCommon/structs.h>
#include <core/segments/core_segment/processing.h>

/**
 * @brief constructor
 */
CoreSegment::CoreSegment()
    : AbstractSegment()
{
    m_type = CORE_SEGMENT;
}

/**
 * @brief constructor to create segment from a snapshot
 *
 * @param data pointer to data with snapshot
 * @param dataSize size of snapshot in number of bytes
 */
CoreSegment::CoreSegment(const void* data, const uint64_t dataSize)
    : AbstractSegment(data, dataSize)
{
    m_type = CORE_SEGMENT;
}

/**
 * @brief destructor
 */
CoreSegment::~CoreSegment() {}

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
 * @brief DynamicSegment::initGpu
 */
void
CoreSegment::initGpu()
{
    Kitsunemimi::ErrorContainer error;

    // create data-object
    data = new Kitsunemimi::GpuData();
    data->numberOfWg.x = 10;
    data->threadsPerWg.x = 10;
    const std::string kernelString(reinterpret_cast<const char*>(gpu_kernel_cl),
                                   gpu_kernel_cl_len);
    if(KyoukoRoot::gpuInterface->addKernel(*data,
                                           "prcessCoreSegment",
                                           kernelString,
                                           error) == false)
    {
        LOG_ERROR(error);
        error._errorMessages.clear();
    }

    if(KyoukoRoot::gpuInterface->addKernel(*data,
                                           "reweightCoreSegment",
                                           kernelString,
                                           error) == false)
    {
        LOG_ERROR(error);
        error._errorMessages.clear();
    }

    assert(data->addBuffer("bricks",                 segmentHeader->bricks.count,             sizeof(Brick),                  false, bricks                    ));
    assert(data->addBuffer("brickOrder",             segmentHeader->brickOrder.count,         sizeof(uint32_t),               false, brickOrder                ));
    assert(data->addBuffer("neuronSections",         segmentHeader->neuronSections.count,     sizeof(NeuronSection),          false, neuronSections            ));
    assert(data->addBuffer("synapseSections",        segmentHeader->synapseSections.count,    sizeof(SynapseSection),         false, synapseSections           ));
    assert(data->addBuffer("segmentHeader",          1,                                       sizeof(SegmentHeader),          false, segmentHeader             ));
    assert(data->addBuffer("segmentSettings",        1,                                       sizeof(SegmentSettings),        false, segmentSettings           ));
    assert(data->addBuffer("inputTransfers",         segmentHeader->inputTransfers.count,     sizeof(float),                  false, inputTransfers            ));
    assert(data->addBuffer("outputTransfers",        segmentHeader->outputTransfers.count,    sizeof(float),                  false, outputTransfers           ));
    assert(data->addBuffer("updatePosSections",      segmentHeader->updatePosSections.count,  sizeof(UpdatePosSection),       false, updatePosSections         ));
    assert(data->addBuffer("randomValues",           NUMBER_OF_RAND_VALUES,                   sizeof(uint32_t),               false, KyoukoRoot::m_randomValues));

    assert(data->addBuffer("sectionConnections",     segmentHeader->synapseSections.count,    sizeof(SectionConnection),      false));
    sectionConnections = static_cast<SectionConnection*>(data->getBufferData("sectionConnections"));
    for(uint32_t i = 0; i < segmentHeader->synapseSections.count; i++) {
        sectionConnections[i] = SectionConnection();
    }

    if(KyoukoRoot::gpuInterface->initCopyToDevice(*data, error) == false) {
        LOG_ERROR(error);
    }

    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "bricks",                 error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "brickOrder",             error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "neuronSections",         error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "sectionConnections",     error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "synapseSections",        error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "updatePosSections",      error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "segmentHeader",          error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "segmentSettings",        error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "inputTransfers",         error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "outputTransfers",        error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "prcessCoreSegment", "randomValues",           error));

    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "bricks",                 error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "brickOrder",             error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "neuronSections",         error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "sectionConnections",     error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "synapseSections",        error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "segmentHeader",          error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "segmentSettings",        error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "inputTransfers",         error));
    assert(KyoukoRoot::gpuInterface->bindKernelToBuffer(*data, "reweightCoreSegment", "outputTransfers",        error));
}

/**
 * @brief initalize segment
 *
 * @param parsedContent json-object with the segment-description
 *
 * @return true, if successful, else false
 */
bool
CoreSegment::initSegment(const std::string &name,
                         const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    uint32_t numberOfNeurons = 0;
    uint32_t numberOfNeuronSections = 0;
    uint32_t totalBorderSize = 0;
    Kitsunemimi::ErrorContainer error;

    // calculate sizes
    uint32_t neuronsInBrick = 0;
    for(uint32_t i = 0; i < segmentMeta.bricks.size(); i++)
    {
        neuronsInBrick = segmentMeta.bricks.at(i).numberOfNeurons;
        numberOfNeurons += neuronsInBrick;
        numberOfNeuronSections += getNumberOfNeuronSections(neuronsInBrick);

        if(segmentMeta.bricks.at(i).type == Kitsunemimi::Hanami::INPUT_BRICK_TYPE
                || segmentMeta.bricks.at(i).type == Kitsunemimi::Hanami::OUTPUT_BRICK_TYPE)
        {
            totalBorderSize += neuronsInBrick;
        }
    }

    // create segment metadata
    const SegmentSettings settings = initSettings(segmentMeta);
    SegmentHeader header = createNewHeader(segmentMeta.bricks.size(),
                                           numberOfNeuronSections,
                                           settings.maxSynapseSections,
                                           totalBorderSize);

    // initialize segment itself
    allocateSegment(header);
    initSegmentPointer(header);
    segmentSettings[0] = settings;

    // init content
    initializeNeurons(segmentMeta);
    addBricksToSegment(segmentMeta);
    connectAllBricks();
    initTargetBrickList();

    // init border
    initSlots(segmentMeta);
    connectBorderBuffer();

    // TODO: check result
    setName(name);

    if(KyoukoRoot::useGpu) {
        initGpu();
    }

    return true;
}

/**
 * @brief DynamicSegment::reinitPointer
 * @return
 */
bool
CoreSegment::reinitPointer(const uint64_t numberOfBytes)
{
    // TODO: checks
    uint8_t* dataPtr = static_cast<uint8_t*>(segmentData.staticData);

    uint64_t pos = 0;
    uint64_t byteCounter = 0;
    segmentHeader = reinterpret_cast<SegmentHeader*>(dataPtr + pos);
    byteCounter += sizeof(SegmentHeader);

    pos = segmentHeader->name.bytePos;
    segmentName = reinterpret_cast<SegmentName*>(dataPtr + pos);
    byteCounter += sizeof(SegmentName);

    pos = segmentHeader->settings.bytePos;
    segmentSettings = reinterpret_cast<SegmentSettings*>(dataPtr + pos);
    byteCounter += sizeof(SegmentSettings);

    pos = segmentHeader->slotList.bytePos;
    segmentSlots = reinterpret_cast<SegmentSlotList*>(dataPtr + pos);
    byteCounter += segmentHeader->slotList.count * sizeof(SegmentSlotList);

    pos = segmentHeader->inputTransfers.bytePos;
    inputTransfers = reinterpret_cast<float*>(dataPtr + pos);
    byteCounter += segmentHeader->inputTransfers.count * sizeof(float);

    pos = segmentHeader->outputTransfers.bytePos;
    outputTransfers = reinterpret_cast<float*>(dataPtr + pos);
    byteCounter += segmentHeader->outputTransfers.count * sizeof(float);

    pos = segmentHeader->bricks.bytePos;
    bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    byteCounter += segmentHeader->bricks.count * sizeof(Brick);

    pos = segmentHeader->brickOrder.bytePos;
    brickOrder = reinterpret_cast<uint32_t*>(dataPtr + pos);
    byteCounter += segmentHeader->brickOrder.count * sizeof(uint32_t);

    pos = segmentHeader->neuronSections.bytePos;
    neuronSections = reinterpret_cast<NeuronSection*>(dataPtr + pos);
    byteCounter += segmentHeader->neuronSections.count * sizeof(NeuronSection);

    pos = segmentHeader->updatePosSections.bytePos;
    updatePosSections = reinterpret_cast<UpdatePosSection*>(dataPtr + pos);
    byteCounter += segmentHeader->updatePosSections.count * sizeof(UpdatePosSection);

    dataPtr = static_cast<uint8_t*>(segmentData.itemData);
    //pos = segmentHeader->synapseSections.bytePos;
    synapseSections = reinterpret_cast<SynapseSection*>(dataPtr);
    byteCounter += segmentHeader->synapseSections.count * sizeof(SynapseSection);

    if(KyoukoRoot::useGpu) {
        initGpu();
    }

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
CoreSegment::initializeNeurons(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    uint32_t sectionPositionOffset = 0;

    for(uint32_t i = 0; i < segmentMeta.bricks.size(); i++)
    {
        int64_t neuronsInBrick = segmentMeta.bricks.at(i).numberOfNeurons;
        const uint32_t numberOfNeuronSectionsInBrick = getNumberOfNeuronSections(neuronsInBrick);

        uint32_t sectionCounter = 0;
        while(sectionCounter < numberOfNeuronSectionsInBrick)
        {
            const uint32_t sectionId = sectionPositionOffset + sectionCounter;
            NeuronSection* section = &neuronSections[sectionId];
            UpdatePosSection* updatePosSection = &updatePosSections[sectionId];

            if(neuronsInBrick >= NEURONS_PER_NEURONSECTION)
            {
                for(uint32_t i = 0; i < NEURONS_PER_NEURONSECTION; i++) {
                    section->neurons[i].border = 0.0f;
                }
                section->numberOfNeurons = NEURONS_PER_NEURONSECTION;
                updatePosSection->numberOfPositions = NEURONS_PER_NEURONSECTION;
                neuronsInBrick -= NEURONS_PER_NEURONSECTION;
            }
            else
            {
                for(uint32_t i = 0; i < neuronsInBrick; i++) {
                    section->neurons[i].border = 0.0f;
                }
                section->numberOfNeurons = neuronsInBrick;
                updatePosSection->numberOfPositions = neuronsInBrick;
                break;
            }
            sectionCounter++;
        }
        sectionPositionOffset += numberOfNeuronSectionsInBrick;
    }

    return true;
}

/**
 * @brief init border-buffer
 *
 * @return true, if successful, else false
 */
bool
CoreSegment::connectBorderBuffer()
{
    NeuronSection* section = nullptr;
    Brick* brick = nullptr;

    uint64_t transferCounter = 0;

    for(uint32_t i = 0; i < segmentHeader->bricks.count; i++)
    {
        brick = &bricks[i];
        if(brick->isInputBrick)
        {
            const uint32_t numberOfNeuronSections = getNumberOfNeuronSections(brick->numberOfNeurons);
            for(uint32_t j = 0; j < numberOfNeuronSections; j++)
            {
                if(transferCounter >= segmentHeader->inputTransfers.count) {
                    break;
                }

                section = &neuronSections[brick->neuronSectionPos + j];
                for(uint32_t k = 0; k < section->numberOfNeurons; k++)
                {
                    section->neurons[k].targetBorderId = transferCounter;
                    transferCounter++;
                }
            }
        }

        // connect output-bricks with border-buffer
        if(brick->isOutputBrick)
        {
            const uint32_t numberOfNeuronSections = getNumberOfNeuronSections(brick->numberOfNeurons);
            for(uint32_t j = 0; j < numberOfNeuronSections; j++)
            {
                if(transferCounter >= segmentHeader->outputTransfers.count) {
                    break;
                }

                section = &neuronSections[brick->neuronSectionPos + j];
                for(uint32_t k = 0; k < section->numberOfNeurons; k++)
                {
                    section->neurons[k].targetBorderId = transferCounter;
                    transferCounter++;
                }
            }
        }
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
CoreSegment::initSettings(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    SegmentSettings settings;

    // parse settings
    settings.synapseSegmentation = segmentMeta.synapseSegmentation;
    settings.signNeg = segmentMeta.signNeg;
    settings.maxSynapseSections = segmentMeta.maxSynapseSections;

    return settings;
}

/**
 * @brief create new segment-header with size and position information
 *
 * @param numberOfBricks number of bricks
 * @param numberOfNeurons number of neurons
 * @param numberOfSynapseSections number of synapse-sections
 * @param borderbufferSize size of border-buffer
 *
 * @return new segment-header
 */
SegmentHeader
CoreSegment::createNewHeader(const uint32_t numberOfBricks,
                             const uint32_t numberOfNeuronSections,
                             const uint64_t numberOfSynapseSections,
                             const uint64_t borderbufferSize)
{
    SegmentHeader segmentHeader;
    segmentHeader.segmentType = m_type;
    uint32_t segmentDataPos = createGenericNewHeader(segmentHeader, borderbufferSize);

    // init bricks
    segmentHeader.bricks.count = numberOfBricks;
    segmentHeader.bricks.bytePos = segmentDataPos;
    segmentDataPos += numberOfBricks * sizeof(Brick);

    // init brick-order
    segmentHeader.brickOrder.count = numberOfBricks;
    segmentHeader.brickOrder.bytePos = segmentDataPos;
    segmentDataPos += numberOfBricks * sizeof(uint32_t);

    // init neurons
    segmentHeader.neuronSections.count = numberOfNeuronSections;
    segmentHeader.neuronSections.bytePos = segmentDataPos;
    segmentDataPos += numberOfNeuronSections * sizeof(NeuronSection);

    // init section-updates
    segmentHeader.updatePosSections.count = numberOfNeuronSections;
    segmentHeader.updatePosSections.bytePos = segmentDataPos;
    segmentDataPos += numberOfNeuronSections * sizeof(UpdatePosSection);

    segmentHeader.staticDataSize = segmentDataPos;

    // init synapse sections
    segmentDataPos = 0;
    segmentHeader.synapseSections.count = numberOfSynapseSections;
    segmentHeader.synapseSections.bytePos = segmentDataPos;

    return segmentHeader;
}

/**
 * @brief init pointer within the segment-header
 *
 * @param header segment-header
 */
void
CoreSegment::initSegmentPointer(const SegmentHeader &header)
{
    uint8_t* dataPtr = static_cast<uint8_t*>(segmentData.staticData);
    uint64_t pos = 0;

    segmentHeader = reinterpret_cast<SegmentHeader*>(dataPtr + pos);
    segmentHeader[0] = header;

    pos = segmentHeader->name.bytePos;
    segmentName = reinterpret_cast<SegmentName*>(dataPtr + pos);

    pos = segmentHeader->settings.bytePos;
    segmentSettings = reinterpret_cast<SegmentSettings*>(dataPtr + pos);

    pos = segmentHeader->slotList.bytePos;
    segmentSlots = reinterpret_cast<SegmentSlotList*>(dataPtr + pos);

    pos = segmentHeader->inputTransfers.bytePos;
    inputTransfers = reinterpret_cast<float*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->inputTransfers.count; i++) {
        inputTransfers[i] = 0.0f;
    }

    pos = segmentHeader->outputTransfers.bytePos;
    outputTransfers = reinterpret_cast<float*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->outputTransfers.count; i++) {
        outputTransfers[i] = 0.0f;
    }

    pos = segmentHeader->bricks.bytePos;
    bricks = reinterpret_cast<Brick*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->bricks.count; i++) {
        bricks[i] = Brick();
    }

    pos = segmentHeader->brickOrder.bytePos;
    brickOrder = reinterpret_cast<uint32_t*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->bricks.count; i++) {
        brickOrder[i] = i;
    }

    pos = segmentHeader->neuronSections.bytePos;
    neuronSections = reinterpret_cast<NeuronSection*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->neuronSections.count; i++) {
        neuronSections[i] = NeuronSection();
    }

    pos = segmentHeader->updatePosSections.bytePos;
    updatePosSections = reinterpret_cast<UpdatePosSection*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->updatePosSections.count; i++) {
        updatePosSections[i] = UpdatePosSection();
    }

    dataPtr = static_cast<uint8_t*>(segmentData.itemData);
    pos = segmentHeader->synapseSections.bytePos;
    synapseSections = reinterpret_cast<SynapseSection*>(dataPtr + pos);
}

/**
 * @brief allocate memory for the segment
 *
 * @param header header with the size-information
 */
void
CoreSegment::allocateSegment(SegmentHeader &header)
{
    segmentData.initBuffer<SynapseSection>(header.synapseSections.count, header.staticDataSize);
    segmentData.deleteAll();
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
CoreSegment::createNewBrick(const Kitsunemimi::Hanami::BrickMeta &brickMeta,
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
    newBrick.numberOfNeuronSections = getNumberOfNeuronSections(brickMeta.numberOfNeurons);

    for(uint8_t side = 0; side < 12; side++) {
        newBrick.neighbors[side] = UNINIT_STATE_32;
    }

    return newBrick;
}

/**
 * @brief init all bricks
 *
 * @param metaBase json with all brick-definitions
 */
void
CoreSegment::addBricksToSegment(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    uint32_t neuronBrickIdCounter = 0;
    uint32_t neuronSectionPosCounter = 0;
    NeuronSection* section = nullptr;
    uint32_t neuronIdCounter = 0;

    for(uint32_t i = 0; i < segmentMeta.bricks.size(); i++)
    {
        Brick newBrick = createNewBrick(segmentMeta.bricks.at(i), i);
        newBrick.neuronSectionPos = neuronSectionPosCounter;

        for(uint32_t j = 0; j < newBrick.numberOfNeuronSections; j++)
        {
            section = &neuronSections[j + neuronSectionPosCounter];
            section->brickId = newBrick.brickId;
            for(uint32_t k = 0; k < section->numberOfNeurons; k++) {
                neuronIdCounter++;
            }
        }

        // copy new brick to segment
        bricks[neuronBrickIdCounter] = newBrick;
        assert(neuronBrickIdCounter == newBrick.brickId);
        neuronBrickIdCounter++;
        neuronSectionPosCounter += newBrick.numberOfNeuronSections;
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
CoreSegment::connectBrick(Brick* sourceBrick,
                          const uint8_t side)
{
    const Kitsunemimi::Hanami::Position next = getNeighborPos(sourceBrick->brickPos, side);
    // debug-output
    // std::cout<<next.x<<" : "<<next.y<<" : "<<next.z<<std::endl;

    if(next.isValid())
    {
        for(uint32_t t = 0; t < segmentHeader->bricks.count; t++)
        {
            Brick* targetBrick = &bricks[t];
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
CoreSegment::connectAllBricks()
{
    for(uint32_t i = 0; i < segmentHeader->bricks.count; i++)
    {
        Brick* sourceBrick = &bricks[i];
        for(uint8_t side = 0; side < 12; side++) {
            connectBrick(sourceBrick, side);
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
CoreSegment::goToNextInitBrick(Brick* currentBrick, uint32_t* maxPathLength)
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
            return goToNextInitBrick(&bricks[nextBrickId], maxPathLength);
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
CoreSegment::initTargetBrickList()
{
    for(uint32_t i = 0; i < segmentHeader->bricks.count; i++)
    {
        Brick* baseBrick = &bricks[i];

        // ignore output-bricks, because they only forward to the border-buffer
        // and not to other bricks
        if(baseBrick->isOutputBrick) {
            continue;
        }

        // test 1000 samples for possible next bricks
        for(uint32_t counter = 0; counter < 1000; counter++)
        {
            uint32_t maxPathLength = 2; // TODO: make configurable
            const uint32_t brickId = goToNextInitBrick(baseBrick, &maxPathLength);
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

/**
 * @brief initialize the border-buffer and neighbor-list of the segment for each side
 *
 * @param segmentTemplate parsend content with the required information
 *
 * @return true, if successful, else false
 */
bool
CoreSegment::initSlots(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    uint64_t posCounter = 0;
    uint32_t slotCounter = 0;

    for(uint32_t i = 0; i < segmentMeta.bricks.size(); i++)
    {
        if(segmentMeta.bricks.at(i).type != Kitsunemimi::Hanami::INPUT_BRICK_TYPE
                && segmentMeta.bricks.at(i).type != Kitsunemimi::Hanami::OUTPUT_BRICK_TYPE)
        {
            continue;
        }

        const uint32_t numberOfNeurons = segmentMeta.bricks.at(i).numberOfNeurons;
        SegmentSlot* currentSlot = &segmentSlots->slots[slotCounter];
        currentSlot->setName(segmentMeta.bricks.at(i).name);
        currentSlot->numberOfNeurons = numberOfNeurons;
        currentSlot->inputTransferBufferPos = posCounter;
        currentSlot->outputTransferBufferPos = posCounter;

        if(segmentMeta.bricks.at(i).type == Kitsunemimi::Hanami::INPUT_BRICK_TYPE) {
            currentSlot->direction = INPUT_DIRECTION;
        } else {
            currentSlot->direction = OUTPUT_DIRECTION;
        }

        // update total position pointer, because all border-buffers are in the same blog
        // beside each other
        posCounter += numberOfNeurons;
        slotCounter++;
    }

    assert(posCounter == segmentHeader->inputTransfers.count);

    return true;
}
