/**
 * @file        input_segment.h
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

#ifndef KYOUKOMIND_INPUT_SEGMENTS_H
#define KYOUKOMIND_INPUT_SEGMENTS_H

#include <common.h>

#include <core/segments/abstract_segment.h>

#include "objects.h"

class InputSegment
        : public AbstractSegment
{
public:
    InputSegment();
    InputSegment(const void* data, const uint64_t dataSize);
    ~InputSegment();

    InputNeuron* inputs = nullptr;

    bool initSegment(const std::string &name,
                     const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);
    bool reinitPointer(const uint64_t numberOfBytes);

private:
    SegmentHeader createNewHeader(const uint32_t numberOfInputs,
                                  const uint64_t borderbufferSize);
    void initSegmentPointer(const SegmentHeader &header);
    bool connectBorderBuffer();
    void allocateSegment(SegmentHeader &header);
    bool initSlots(const uint32_t numberOfInputs);
};

#endif // KYOUKOMIND_INPUT_SEGMENTS_H
