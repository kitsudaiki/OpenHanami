/**
 * @file        input_segment.cpp
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

#include "input_segment.h"

/**
 * @brief constructor
 */
InputSegment::InputSegment()
    : AbstractSegment()
{
    m_type = INPUT_SEGMENT;
}

/**
 * @brief constructor to create segment from a snapshot
 *
 * @param data pointer to data with snapshot
 * @param dataSize size of snapshot in number of bytes
 */
InputSegment::InputSegment(const void* data, const uint64_t dataSize)
    : AbstractSegment(data, dataSize)
{
    m_type = INPUT_SEGMENT;
}

/**
 * @brief destructor
 */
InputSegment::~InputSegment() {}

/**
 * @brief initalize segment
 *
 * @param parsedContent json-object with the segment-description
 *
 * @return true, if successful, else false
 */
bool
InputSegment::initSegment(const std::string &name,
                          const Kitsunemimi::Hanami::SegmentMeta &segmentMeta)
{
    const uint32_t numberOfInputs = segmentMeta.bricks.at(0).numberOfNeurons;
    const uint32_t totalBorderSize = numberOfInputs;

    SegmentHeader header = createNewHeader(numberOfInputs, totalBorderSize);

    allocateSegment(header);
    initSegmentPointer(header);
    connectBorderBuffer();

    initSlots(numberOfInputs);

    // TODO: check result
    setName(name);

    return true;
}

/**
 * @brief InputSegment::reinitPointer
 * @return
 */
bool
InputSegment::reinitPointer(const uint64_t numberOfBytes)
{
    uint8_t* dataPtr = static_cast<uint8_t*>(segmentData.data);

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

    pos = segmentHeader->inputs.bytePos;
    inputs = reinterpret_cast<InputNeuron*>(dataPtr + pos);
    byteCounter += segmentHeader->inputs.count * sizeof(InputNeuron);

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
InputSegment::connectBorderBuffer()
{
    for(uint32_t i = 0; i < segmentHeader->inputs.count; i++) {
        inputs[i].targetBorderId = i;
    }

    return true;
}

/**
 * @brief create new segment-header with size and position information
 *
 * @param numberOfInputs number of inputs
 * @param borderbufferSize size of border-buffer
 *
 * @return new segment-header
 */
SegmentHeader
InputSegment::createNewHeader(const uint32_t numberOfInputs,
                              const uint64_t borderbufferSize)
{
    SegmentHeader segmentHeader;
    segmentHeader.segmentType = m_type;
    uint32_t segmentDataPos = createGenericNewHeader(segmentHeader, borderbufferSize);

    // init bricks
    segmentHeader.inputs.count = numberOfInputs;
    segmentHeader.inputs.bytePos = segmentDataPos;
    segmentDataPos += numberOfInputs * sizeof(InputNeuron);

    segmentHeader.staticDataSize = segmentDataPos;

    return segmentHeader;
}

/**
 * @brief init pointer within the segment-header
 *
 * @param header segment-header
 */
void
InputSegment::initSegmentPointer(const SegmentHeader &header)
{
    uint8_t* dataPtr = static_cast<uint8_t*>(segmentData.data);
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

    pos = segmentHeader->inputs.bytePos;
    inputs = reinterpret_cast<InputNeuron*>(dataPtr + pos);
    for(uint32_t i = 0; i < segmentHeader->inputs.count; i++) {
        inputs[i] = InputNeuron();
    }
}

/**
 * @brief allocate memory for the segment
 *
 * @param header header with the size-information
 */
void
InputSegment::allocateSegment(SegmentHeader &header)
{
    Kitsunemimi::reset_DataBuffer(segmentData, Kitsunemimi::calcBytesToBlocks(header.staticDataSize));
}

/**
 * @brief initialize the slots
 *
 * @param numberOfInputs number of inputs
 *
 * @return true, if successful, else false
 */
bool
InputSegment::initSlots(const uint32_t numberOfInputs)
{
    for(uint32_t i = 0; i < 16; i++)
    {
        const uint32_t size  = numberOfInputs;

        // init new segment-neighbor
        SegmentSlot* currentSlot = &segmentSlots->slots[i];
        currentSlot->setName("output");
        currentSlot->inUse = false;
        currentSlot->numberOfNeurons = size;
        currentSlot->inputTransferBufferPos = 0;
        currentSlot->outputTransferBufferPos = 0;
        currentSlot->direction = OUTPUT_DIRECTION;
    }

    return true;
}
