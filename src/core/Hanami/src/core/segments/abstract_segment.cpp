/**
 * @file        abstract_segment.cpp
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

#include "abstract_segment.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 */
AbstractSegment::AbstractSegment() {}

AbstractSegment::AbstractSegment(const void* data, const uint64_t dataSize)
{
    segmentData.initBuffer(data, dataSize);
}

/**
 * @brief destructor
 */
AbstractSegment::~AbstractSegment() {}

/**
 * @brief get type of the segment
 *
 * @return type of the segment
 */
SegmentTypes
AbstractSegment::getType() const
{
    return m_type;
}

/**
 * @brief get name of the segment
 *
 * @return name of the segment
 */
const std::string
AbstractSegment::getName() const
{
    return segmentName->getName();
}

/**
 * @brief set new name for the segment
 *
 * @param name new name
 *
 * @return false, if name is too long or empty, else true
 */
bool
AbstractSegment::setName(const std::string &name)
{
    return segmentName->setName(name);
}

/**
 * @brief AbstractSegment::getSlotId
 * @param name
 * @return
 */
uint8_t
AbstractSegment::getSlotId(const std::string &name)
{
    for(uint64_t i = 0; i < 16; i++)
    {
        if(segmentSlots->slots[i].getName() == name) {
            return i;
        }
    }

    return UNINIT_STATE_8;
}

/**
 * @brief check if all border-buffer, which are in use, are ready for processing
 *
 * @return true, if all border-buffer are ready, else false
 */
bool
AbstractSegment::isReady()
{
    for(uint8_t i = 0; i < 16; i++)
    {
        if(segmentSlots->slots[i].inUse == true
                && segmentSlots->slots[i].inputReady == false)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief run finishing step of the segment-processing to share the border-buffer with the
 *        neighbor segments
 */
void
AbstractSegment::finishSegment()
{
    float* sourceBuffer = nullptr;
    float* targetBuffer  = nullptr;
    uint32_t targetId = 0;
    uint8_t targetSide = 0;
    uint64_t targetBufferPos = 0;
    AbstractSegment* targetSegment = nullptr;
    SegmentSlotList* targetNeighbors = nullptr;

    for(uint8_t i = 0; i < 16; i++)
    {
        if(segmentSlots->slots[i].inUse == 1)
        {
            // get information of the neighbor
            sourceBuffer = &outputTransfers[segmentSlots->slots[i].outputTransferBufferPos];
            targetId = segmentSlots->slots[i].targetSegmentId;
            targetSide = segmentSlots->slots[i].targetSlotId;

            // copy data to the target buffer and wipe the source buffer
            targetSegment = parentCluster->allSegments.at(targetId);
            targetNeighbors = targetSegment->segmentSlots;
            targetBufferPos = targetNeighbors->slots[targetSide].inputTransferBufferPos;
            targetBuffer = &targetSegment->inputTransfers[targetBufferPos];
            memcpy(targetBuffer,
                   sourceBuffer,
                   segmentSlots->slots[i].numberOfNeurons * sizeof(float));
            memset(sourceBuffer,
                   0,
                   segmentSlots->slots[i].numberOfNeurons * sizeof(float));

            // mark the target as ready for processing
            targetSegment->segmentSlots->slots[targetSide].inputReady = true;
        }
    }

    parentCluster->updateClusterState();
}

/**
 * @brief generate header with generic segment-information
 *
 * @param header reference to the header-object to fill
 * @param borderbufferSize size of the border-buffer in bytes
 *
 * @return number of required bytes to the generic information
 */
uint32_t
AbstractSegment::createGenericNewHeader(SegmentHeader &header,
                                        const uint64_t borderbufferSize)
{
    uint32_t segmentDataPos = 0;

    // init header
    segmentDataPos += sizeof(SegmentHeader);

    // init name
    header.name.count = 1;
    header.name.bytePos = segmentDataPos;
    segmentDataPos += sizeof(SegmentName);

    // init settings
    header.settings.count = 1;
    header.settings.bytePos = segmentDataPos;
    segmentDataPos += sizeof(SegmentSettings);

    // init neighborList
    header.slotList.count = 1;
    header.slotList.bytePos = segmentDataPos;
    segmentDataPos += sizeof(SegmentSlotList);

    // init inputTransfers
    header.inputTransfers.count = borderbufferSize;
    header.inputTransfers.bytePos = segmentDataPos;
    segmentDataPos += borderbufferSize * sizeof(float);

    // init outputTransfers
    header.outputTransfers.count = borderbufferSize;
    header.outputTransfers.bytePos = segmentDataPos;
    segmentDataPos += borderbufferSize * sizeof(float);

    return segmentDataPos;
}
