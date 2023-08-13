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
 * @brief generate header with generic segment-information
 *
 * @param header reference to the header-object to fill
 * @param borderbufferSize size of the border-buffer in bytes
 *
 * @return number of required bytes to the generic information
 */
uint32_t
AbstractSegment::createGenericNewHeader(SegmentHeader &header,
                                        const uint64_t inputSize,
                                        const uint64_t outputSize)
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

    // init inputTransfers
    header.inputValues.count = inputSize;
    header.inputValues.bytePos = segmentDataPos;
    segmentDataPos += inputSize * sizeof(float);

    // init outputTransfers
    header.outputValues.count = outputSize;
    header.outputValues.bytePos = segmentDataPos;
    segmentDataPos += outputSize * sizeof(float);

    // init outputTransfers
    header.expectedValues.count = outputSize;
    header.expectedValues.bytePos = segmentDataPos;
    segmentDataPos += outputSize * sizeof(float);

    return segmentDataPos;
}
