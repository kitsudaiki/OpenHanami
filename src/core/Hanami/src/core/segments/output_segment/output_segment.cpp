/**
 * @file        output_segment.cpp
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

#include "output_segment.h"

/**
 * @brief constructor
 */
OutputSegment::OutputSegment()
    : AbstractSegment()
{
    m_type = OUTPUT_SEGMENT;
}

/**
 * @brief constructor to create segment from a snapshot
 *
 * @param data pointer to data with snapshot
 * @param dataSize size of snapshot in number of bytes
 */
OutputSegment::OutputSegment(const void* data, const uint64_t dataSize)
    : AbstractSegment(data, dataSize)
{
    m_type = OUTPUT_SEGMENT;
}

/**
 * @brief destructor
 */
OutputSegment::~OutputSegment() {}

/**
 * @brief initalize segment
 *
 * @param parsedContent json-object with the segment-description
 *
 * @return true, if successful, else false
 */
bool
OutputSegment::initSegment(const std::string &name,
                           const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    const uint32_t numberOfOutputs = segmentMeta.bricks.at(0).numberOfNeurons;
    const uint32_t totalBorderSize = numberOfOutputs;

    SegmentHeader header = createNewHeader(numberOfOutputs, totalBorderSize);

    allocateSegment(header);
    initSegmentPointer(header);
    connectBorderBuffer();

    initSlots(numberOfOutputs);

    // TODO: check result
    setName(name);

    return true;
}

/**
 * @brief OutputSegment::reinitPointer
 * @return
 */
bool
OutputSegment::reinitPointer(const uint64_t numberOfBytes)
{
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

    pos = segmentHeader->outputs.bytePos;
    outputs = reinterpret_cast<OutputNeuron*>(dataPtr + pos);
    byteCounter += segmentHeader->outputs.count * sizeof(OutputNeuron);

    // check result
    if(byteCounter != numberOfBytes) {
        return false;
    }

    return true;
}

/**
 * @brief init border-buffer
 *
 * @return true, if successful, else false
 */
bool
OutputSegment::connectBorderBuffer()
{
    for(uint32_t i = 0; i < segmentHeader->outputs.count; i++) {
        outputs[i].targetBorderId = i;
    }

    return true;
}

/**
 * @brief create new segment-header with size and position information
 *
 * @param numberOfOutputs number of outputs
 * @param borderbufferSize size of border-buffer
 *
 * @return new segment-header
 */
SegmentHeader
OutputSegment::createNewHeader(const uint32_t numberOfOutputs,
                               const uint64_t borderbufferSize)
{
    SegmentHeader segmentHeader;
    segmentHeader.segmentType = m_type;
    uint32_t segmentDataPos = createGenericNewHeader(segmentHeader, borderbufferSize);

    // init outputs
    segmentHeader.outputs.count = numberOfOutputs;
    segmentHeader.outputs.bytePos = segmentDataPos;
    segmentDataPos += numberOfOutputs * sizeof(OutputNeuron);

    // set total size of the segment
    segmentHeader.staticDataSize = segmentDataPos;

    return segmentHeader;
}

/**
 * @brief init pointer within the segment-header
 *
 * @param header segment-header
 */
void
OutputSegment::initSegmentPointer(const SegmentHeader &header)
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
    std::fill_n(inputTransfers, segmentHeader->inputTransfers.count, 0.0f);

    pos = segmentHeader->outputTransfers.bytePos;
    outputTransfers = reinterpret_cast<float*>(dataPtr + pos);
    std::fill_n(outputTransfers, segmentHeader->outputTransfers.count, 0.0f);

    pos = segmentHeader->outputs.bytePos;
    outputs = reinterpret_cast<OutputNeuron*>(dataPtr + pos);
    std::fill_n(outputs, segmentHeader->outputs.count, OutputNeuron());
}

/**
 * @brief allocate memory for the segment
 *
 * @param header header with the size-information
 */
void
OutputSegment::allocateSegment(SegmentHeader &header)
{
    const uint32_t numberOfBlocks = (header.staticDataSize / 4096) + 1;
    header.staticDataSize = numberOfBlocks * 4096;
    segmentData.initBuffer(header.staticDataSize);
}

/**
 * @brief initialize the slots
 *
 * @param numberOfInputs number of inputs
 *
 * @return true, if successful, else false
 */
bool
OutputSegment::initSlots(const uint32_t numberOfInputs)
{
    for(uint32_t i = 0; i < 1; i++)
    {
        const uint32_t size  = numberOfInputs;

        // init new segment-neighbor
        SegmentSlot* currentSlot = &segmentSlots->slots[i];
        currentSlot->setName("input");
        currentSlot->inUse = false;
        currentSlot->numberOfNeurons = size;
        currentSlot->inputTransferBufferPos = 0;
        currentSlot->outputTransferBufferPos = 0;
        currentSlot->direction = INPUT_DIRECTION;
    }

    return true;
}
