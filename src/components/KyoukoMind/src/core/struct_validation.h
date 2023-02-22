/**
 * @file        struct_validation.h
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

#ifndef KYOUKOMIND_STRUCT_VALIDATION_H
#define KYOUKOMIND_STRUCT_VALIDATION_H

#include <core/segments/abstract_segment.h>
#include <core/cluster/cluster.h>
#include <core/segments/brick.h>
#include <core/segments/core_segment/objects.h>

/**
 * @brief validate to sized of all structs to ensure, that they have all the defined size of 2^x
 */
inline void
validateStructSizes()
{
    assert(sizeof(SegmentHeader) == 512);
    assert(sizeof(SegmentName) == 256);
    assert(sizeof(Brick) == 4096);
    assert(sizeof(BrickHeader) == 32);
    assert(sizeof(SynapseSection) == 512);
    assert(sizeof(Synapse) == 16);
    assert(sizeof(UpdatePosSection) == 1024);
    assert(sizeof(UpdatePos) == 16);
    assert(sizeof(NeuronSection) == 2048);
    assert(sizeof(Neuron) == 32);

    assert(sizeof(SegmentSlot) == 64);
    assert(sizeof(SegmentSlotList) == 1024);
    assert(sizeof(Cluster::MetaData) == 2048);
    assert(sizeof(Cluster::Settings) == 256);
    assert(sizeof(SegmentSettings) == 256);
    assert(sizeof(Kitsunemimi::Hanami::kuuid) == 40);
}

#endif // KYOUKOMIND_STRUCT_VALIDATION_H
