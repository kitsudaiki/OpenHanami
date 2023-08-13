/**
 * @file        abstract_segment.h
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

#ifndef HANAMI_ABSTRACT_SEGMENTS_H
#define HANAMI_ABSTRACT_SEGMENTS_H

#include <common.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCommon/buffer/item_buffer.h>

#include <libKitsunemimiHanamiClusterParser/segment_meta.h>

#include <core/segments/segment_meta.h>
#include <core/segments/core_segment/objects.h>

class Cluster;

class AbstractSegment
{
public:
    AbstractSegment();
    AbstractSegment(const void* data, const uint64_t dataSize);
    virtual ~AbstractSegment();

    SegmentTypes getType() const;
    const std::string getName() const;
    bool setName(const std::string &name);

    Kitsunemimi::ItemBuffer segmentData;

    SegmentHeader* segmentHeader = nullptr;
    SegmentSettings* segmentSettings = nullptr;

    SegmentName* segmentName = nullptr;
    Cluster* parentCluster = nullptr;

protected:
    SegmentTypes m_type = UNDEFINED_SEGMENT;

    uint32_t createGenericNewHeader(SegmentHeader &header,
                                    const uint64_t inputSize,
                                    const uint64_t outputSize);
};

//==================================================================================================

#endif // HANAMI_ABSTRACT_SEGMENTS_H
