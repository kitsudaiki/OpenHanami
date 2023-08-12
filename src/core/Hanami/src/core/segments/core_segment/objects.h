/**
 * @file        objects.h
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

#ifndef HANAMI_CORE_SEGMENT_OBJECTS_H
#define HANAMI_CORE_SEGMENT_OBJECTS_H

#include <stdint.h>
#include <cstdlib>
#include <algorithm>
#include <libKitsunemimiCommon/structs.h>

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8 0xFF

#define UNINTI_POINT_32 0x0FFFFFFF

// network-predefines
#define SYNAPSES_PER_SYNAPSESECTION 64
#define NUMBER_OF_SYNAPSESECTION 64
#define NEURONS_PER_NEURONSECTION 63
#define POSSIBLE_NEXT_AXON_STEP 80
#define NEURON_CONNECTIONS 512

// processing
#define NUMBER_OF_PROCESSING_UNITS 1
#define NUMBER_OF_RAND_VALUES 10485760

//==================================================================================================

struct Brick
{
    // common
    uint32_t brickId = UNINIT_STATE_32;
    bool isOutputBrick = false;
    bool isInputBrick = false;
    uint8_t padding1[14];
    uint32_t brickBlockPos = UNINIT_STATE_32;

    uint32_t numberOfNeurons = 0;
    uint32_t numberOfNeuronSections = 0;

    Kitsunemimi::Position brickPos;
    uint32_t neighbors[12];
    uint32_t possibleTargetNeuronBrickIds[1000];
};
static_assert(sizeof(Brick) == 4096);

//==================================================================================================

struct Synapse
{
    float weight = 0.0f;
    float border = 0.0f;
    uint16_t targetNeuronId = UNINIT_STATE_16;
    int8_t activeCounter = 0;
    uint8_t padding[5];
};
static_assert(sizeof(Synapse) == 16);

//==================================================================================================

struct SynapseBlock
{
    Synapse synapses[NUMBER_OF_SYNAPSESECTION][SYNAPSES_PER_SYNAPSESECTION];

    SynapseBlock()
    {
        for(uint32_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            std::fill_n(synapses[i], SYNAPSES_PER_SYNAPSESECTION, Synapse());
        }
    }
};
static_assert(sizeof(SynapseBlock) == 64*1024);

//==================================================================================================

struct LocationPtr
{
    uint32_t blockId = UNINIT_STATE_32;
    uint16_t sectionId = UNINIT_STATE_16;
    bool isNeuron = false;
    uint8_t padding[1];
};
static_assert(sizeof(LocationPtr) == 8);

//==================================================================================================

struct Neuron
{
    float input = 0.0f;
    float border = 100.0f;
    float potential = 0.0f;
    float delta = 0.0f;

    uint8_t refractionTime = 1;
    uint8_t active = 0;
    uint8_t padding[2];
    LocationPtr target;
    uint32_t targetBorderId = UNINIT_STATE_32;
};
static_assert(sizeof(Neuron) == 32);

//==================================================================================================

struct NeuronBlock
{
    uint64_t triggerMap;
    uint64_t triggerCompare;
    uint32_t numberOfNeurons = 0;
    uint32_t brickId = 0;
    uint32_t randomPos = 0;
    uint32_t backwardNextId = UNINIT_STATE_32;

    Neuron neurons[NEURONS_PER_NEURONSECTION];

    NeuronBlock()
    {
        randomPos = rand();
        std::fill_n(neurons, NEURONS_PER_NEURONSECTION, Neuron());
    }
};
static_assert(sizeof(NeuronBlock) == 2048);

//==================================================================================================

struct SynapseConnection
{
    uint8_t active = 1;
    uint8_t padding[3];
    uint32_t targetNeuronBlockId = UNINIT_STATE_32;
    uint32_t backwardNextId = UNINIT_STATE_32;

    LocationPtr next[NUMBER_OF_SYNAPSESECTION];
    LocationPtr origin[NUMBER_OF_SYNAPSESECTION];
    float offset[NUMBER_OF_SYNAPSESECTION];

    SynapseConnection()
    {
        std::fill_n(next, NUMBER_OF_SYNAPSESECTION, LocationPtr());
        std::fill_n(origin, NUMBER_OF_SYNAPSESECTION, LocationPtr());
        std::fill_n(offset, NUMBER_OF_SYNAPSESECTION, 0.0f);
    }
};
static_assert(sizeof(SynapseConnection) == 1292);

//==================================================================================================

struct SegmentSettings
{
    uint64_t maxSynapseSections = 0;
    float synapseDeleteBorder = 1.0f;
    float neuronCooldown = 100.0f;
    float memorizing = 0.1f;
    float gliaValue = 1.0f;
    float signNeg = 0.6f;
    float potentialOverflow = 1.0f;
    float synapseSegmentation = 10.0f;
    float backpropagationBorder = 0.00001f;
    uint8_t refractionTime = 1;
    uint8_t doLearn = 0;
    uint8_t updateSections = 0;

    uint8_t padding[213];
};
static_assert(sizeof(SegmentSettings) == 256);

//==================================================================================================

struct SegmentSizes
{
    // synapse-segment
    uint32_t numberOfInputTransfers = 0;
    uint32_t numberOfOutputTransfers = 0;
    uint32_t numberOfBricks = 0;
    uint32_t numberOfInputs = 0;
    uint32_t numberOfOutputs = 0;
    uint32_t numberOfNeuronConnections = 0;
    uint32_t numberOfNeuronSections = 0;
    uint32_t numberOfSynapseSections = 0;
};
static_assert(sizeof(SegmentSizes) == 32);

#endif // HANAMI_CORE_SEGMENT_OBJECTS_H
