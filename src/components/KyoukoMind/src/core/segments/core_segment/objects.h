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

#ifndef KYOUKOMIND_CORE_SEGMENT_OBJECTS_H
#define KYOUKOMIND_CORE_SEGMENT_OBJECTS_H

#include <stdint.h>

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8 0xFF

#define UNINTI_POINT_32 0x0FFFFFFF

// network-predefines
#define SYNAPSES_PER_SYNAPSESECTION 30
#define NEURONS_PER_NEURONSECTION 63
#define POSSIBLE_NEXT_AXON_STEP 80
#define NEURON_CONNECTIONS 512

// processing
#define NUMBER_OF_PROCESSING_UNITS 1
#define NUMBER_OF_RAND_VALUES 10485760

//==================================================================================================

struct BrickHeader
{
    // common
    uint32_t brickId = UNINIT_STATE_32;
    bool isOutputBrick = false;
    bool isInputBrick = false;
    uint8_t padding1[14];
    uint32_t neuronSectionPos = UNINIT_STATE_32;

    uint32_t numberOfNeurons = 0;
    uint32_t numberOfNeuronSections = 0;

    // total size: 32 Bytes
};

//==================================================================================================

struct Neuron
{
    float input = 0.0f;
    float border = 100.0f;
    float potential = 0.0f;
    float delta = 0.0f;

    uint8_t refractionTime = 1;
    uint8_t active = 0;
    uint8_t padding[6];

    uint32_t targetBorderId = UNINIT_STATE_32;
    uint32_t targetSectionId = UNINIT_STATE_32;

    // total size: 32 Byte
};

//==================================================================================================

struct NeuronSection
{
    Neuron neurons[NEURONS_PER_NEURONSECTION];
    uint32_t numberOfNeurons = 0;
    uint32_t brickId = 0;
    uint8_t padding[24];

    NeuronSection()
    {
        for(uint32_t i = 0; i < NEURONS_PER_NEURONSECTION; i++) {
            neurons[i] = Neuron();
        }
    }
    // total size: 2048 Byte
};

//==================================================================================================

struct Synapse
{
    float weight = 0.0f;
    float border = 0.0f;
    uint16_t targetNeuronId = UNINIT_STATE_16;
    int8_t activeCounter = 0;
    uint8_t padding[5];
    // total size: 16 Byte
};

//==================================================================================================

struct SynapseConnection
{
    uint8_t active = 1;
    uint8_t padding[3];

    float offset = 0.0f;
    uint32_t randomPos = 0;

    uint32_t forwardNextId = UNINIT_STATE_32;
    uint32_t backwardNextId = UNINIT_STATE_32;

    uint32_t targetNeuronSectionId = UNINIT_STATE_32;
    uint32_t sourceNeuronSectionId = UNINIT_STATE_32;
    uint32_t sourceNeuronId = UNINIT_STATE_32;

    // total size: 32 Byte
};

//==================================================================================================

struct SynapseSection
{
    SynapseConnection connection = SynapseConnection();

    Synapse synapses[SYNAPSES_PER_SYNAPSESECTION];

    SynapseSection()
    {
        for(uint32_t i = 0; i < SYNAPSES_PER_SYNAPSESECTION; i++) {
            synapses[i] = Synapse();
        }
    }
    // total size: 512 Byte
};

//==================================================================================================

struct UpdatePos
{
    uint32_t type = 0;
    uint32_t randomPos = UNINIT_STATE_32;
    float offset = 0.0f;
    uint8_t padding[4];
    // total size: 16 Byte
};

//==================================================================================================

struct UpdatePosSection
{
    UpdatePos positions[NEURONS_PER_NEURONSECTION];
    uint32_t numberOfPositions = 0;
    uint8_t padding[12];

    UpdatePosSection()
    {
        for(uint32_t i = 0; i < NEURONS_PER_NEURONSECTION; i++) {
            positions[i] = UpdatePos();
        }
    }
    // total size: 1024 Byte
};

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

    // total size: 256 Byte
};

struct TempObj
{
    float values[512][64];
    // total size: 128 KiByte
};

//==================================================================================================

struct NeuronConnection
{
    uint32_t backwardIds[512];

    NeuronConnection()
    {
        for(uint32_t i = 0; i < 512; i++) {
            backwardIds[i] = UNINIT_STATE_32;
        }
    }
    // total size: 2048 Byte
};

//==================================================================================================

struct SegmentSizes
{
    // synapse-segment
    uint32_t numberOfInputTransfers = 0;
    uint32_t numberOfOutputTransfers = 0;
    uint32_t numberOfBricks = 0;
    uint32_t numberOfInputs = 0;
    uint32_t numberOfOutputs = 0;
    uint32_t numberOfUpdatePosSections = 0;
    uint32_t numberOfNeuronSections = 0;
    uint32_t numberOfSynapseSections = 0;
    // total size: 32 Byte
};

#endif // KYOUKOMIND_CORE_SEGMENT_OBJECTS_H
