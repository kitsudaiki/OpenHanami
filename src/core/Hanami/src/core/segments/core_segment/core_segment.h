/**
 * @file        core_segment.h
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

#ifndef HANAMI_CORE_SEGMENTS_H
#define HANAMI_CORE_SEGMENTS_H

#include <common.h>

#include <core/segments/abstract_segment.h>
#include "objects.h"

namespace Kitsunemimi {
class GpuData;
}

//==================================================================================================

struct PointerHandler
{
    SegmentSettings* segmentSettings = nullptr;
    uint32_t* randomValues = nullptr;

    float* inputTransfers = nullptr;
    float* outputTransfers = nullptr;
};

class CoreSegment
        : public AbstractSegment
{
public:
    CoreSegment();
    CoreSegment(const void* data, const uint64_t dataSize);
    ~CoreSegment();

    bool initSegment(const std::string &name,
                     const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
    bool reinitPointer(const uint64_t numberOfBytes);

    Brick* bricks = nullptr;
    uint32_t* brickOrder = nullptr;

    BrickBlock* brickBlocks = nullptr;
    BlockConnection* blockConnections = nullptr;

    SegmentSizes segmentSizes;

    uint32_t numberOfBrickBlocks = 0;
    PointerHandler gpuPointer;

    Kitsunemimi::GpuData* data = nullptr;

private:
    SegmentSettings initSettings(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
    SegmentHeader createNewHeader(const uint32_t numberOfBricks,
                                  const uint32_t numberOfBrickBlocks,
                                  const uint64_t borderbufferSize);
    void initSegmentPointer(const SegmentHeader &header);
    bool connectBorderBuffer();
    void allocateSegment(SegmentHeader &header);
    //void initOpencl(Kitsunemimi::GpuData &data);
    void initCuda();

    void addBricksToSegment(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
    bool initTargetBrickList();

    Brick createNewBrick(const Kitsunemimi::Hanami::BrickMeta &brickMeta, const uint32_t id);
    void connectBrick(Brick *sourceBrick, const uint8_t side);
    void connectAllBricks();
    bool initializeNeurons(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
    uint32_t goToNextInitBrick(Brick* currentBrick, uint32_t* maxPathLength);
    bool initSlots(const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
};

#endif // HANAMI_CORE_SEGMENTS_H
