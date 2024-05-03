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

#include <hanami_common/structs.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <uuid/uuid.h>

#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8 0xFF

#define UNINTI_POINT_32 0x0FFFFFFF

// network-predefines
#define SYNAPSES_PER_SYNAPSESECTION 63
#define NUMBER_OF_SYNAPSESECTION 64
#define NEURONS_PER_NEURONBLOCK 64
#define POSSIBLE_NEXT_AXON_STEP 80
#define NUMBER_OF_POSSIBLE_NEXT 90
#define NUMBER_OF_OUTPUT_CONNECTIONS 7

//==================================================================================================

enum ClusterProcessingMode {
    NORMAL_MODE = 0,
    TRAIN_FORWARD_MODE = 1,
    TRAIN_BACKWARD_MODE = 2,
    REDUCTION_MODE = 3,
};

//==================================================================================================

struct kuuid {
    char uuid[UUID_STR_LEN];
    uint8_t padding[3];

    const std::string toString() const { return std::string(uuid, UUID_STR_LEN - 1); }
};
static_assert(sizeof(kuuid) == 40);

//==================================================================================================

struct HeaderEntry {
    uint64_t bytePos = 0;
    uint64_t count = 0;
};
static_assert(sizeof(HeaderEntry) == 16);

//==================================================================================================

struct ClusterSettings {
    float backpropagationBorder = 0.001f;
    float potentialOverflow = 1.0f;

    float neuronCooldown = 1000000000.0f;
    uint32_t refractoryTime = 1;
    uint32_t maxConnectionDistance = 1;
    bool enableReduction = false;

    uint8_t padding[43];
};
static_assert(sizeof(ClusterSettings) == 64);

//==================================================================================================

struct ClusterHeader {
    uint8_t objectType = 0;
    uint8_t version = 1;
    uint8_t padding[2];

    char name[256];
    uint32_t nameSize = 0;

    uint64_t staticDataSize = 0;
    kuuid uuid;

    ClusterSettings settings;

    uint8_t padding2[136];
};
static_assert(sizeof(ClusterHeader) == 512);

//==================================================================================================

struct Synapse {
    float weight = 0.0f;
    float border = 0.0f;

    uint8_t padding2[6];
    uint8_t activeCounter = 0;
    uint8_t targetNeuronId = UNINIT_STATE_8;
};
static_assert(sizeof(Synapse) == 16);

//==================================================================================================

struct SynapseSection {
    Synapse synapses[SYNAPSES_PER_SYNAPSESECTION];
    float tollerance = 0.49f;

    uint8_t padding[11];
    bool hasNext = false;

    SynapseSection() { std::fill_n(synapses, SYNAPSES_PER_SYNAPSESECTION, Synapse()); }
};
static_assert(sizeof(SynapseSection) == 1024);

//==================================================================================================

struct SynapseBlock {
    SynapseSection sections[NUMBER_OF_SYNAPSESECTION];
    float tempValues[NEURONS_PER_NEURONBLOCK];

    SynapseBlock()
    {
        std::fill_n(sections, NUMBER_OF_SYNAPSESECTION, SynapseSection());
        std::fill_n(tempValues, NEURONS_PER_NEURONBLOCK, 0.0f);
    }
};
static_assert(sizeof(SynapseBlock) == 64 * 1024 + 256);

//==================================================================================================

struct SourceLocationPtr {
    // HINT (kitsudaiki): not initialized here, because they are used in shared memory in cuda
    //                    which doesn't support initializing of the values, when defining the
    //                    shared-memory-object
    uint16_t brickId;
    uint16_t blockId;
    uint16_t neuronId;
    uint8_t posInNeuron;
    bool isInput;
};
static_assert(sizeof(SourceLocationPtr) == 8);

//==================================================================================================

struct OutputTargetLocationPtr {
    float connectionWeight = 0.0f;
    uint16_t blockId = UNINIT_STATE_16;
    uint16_t neuronId = UNINIT_STATE_16;
    uint8_t padding[6];
};
static_assert(sizeof(OutputTargetLocationPtr) == 16);

//==================================================================================================

struct Neuron {
    float input = 0.0f;
    float border = 0.0f;
    float potential = 0.0f;
    float delta = 0.0f;

    uint8_t refractoryTime = 1;
    uint8_t active = 0;

    float newLowerBound = 0.0f;
    float potentialRange = 0.0f;
    uint8_t isNew = 0;
    uint8_t inUse = 0;

    void setInUse(const uint8_t pos) { inUse |= (1 << pos); }

    void deleteInUse(const uint8_t pos) { inUse &= (~(1 << pos)); }

    uint8_t getFirstZeroBit()
    {
        for (int i = 0; i < 8; ++i) {
            if ((inUse & (1 << i)) == 0) {
                return i;
            }
        }

        return UNINIT_STATE_8;
    }
};
static_assert(sizeof(Neuron) == 32);

//==================================================================================================

struct NeuronBlock {
    Neuron neurons[NEURONS_PER_NEURONBLOCK];

    NeuronBlock() { std::fill_n(neurons, NEURONS_PER_NEURONBLOCK, Neuron()); }
};
static_assert(sizeof(NeuronBlock) == 2048);

//==================================================================================================

struct TempNeuron {
    float delta[8];

    TempNeuron() { std::fill_n(delta, 8, 0.0f); }
};
static_assert(sizeof(TempNeuron) == 32);

//==================================================================================================

struct TempNeuronBlock {
    TempNeuron neurons[NEURONS_PER_NEURONBLOCK];

    TempNeuronBlock() { std::fill_n(neurons, NEURONS_PER_NEURONBLOCK, TempNeuron()); }
};
static_assert(sizeof(TempNeuronBlock) == 2048);

//==================================================================================================

struct OutputNeuron {
    OutputTargetLocationPtr targets[NUMBER_OF_OUTPUT_CONNECTIONS];
    float outputVal = 0.0f;
    float exprectedVal = 0.0f;
    uint8_t padding[8];
};
static_assert(sizeof(OutputNeuron) == 128);

//==================================================================================================

struct InputNeuron {
    uint32_t neuronId = UNINIT_STATE_32;
    float value = 0.0f;
};
static_assert(sizeof(InputNeuron) == 8);

//==================================================================================================

struct OutputInterface {
    uint32_t targetBrickId = UNINIT_STATE_32;
    uint32_t numberOfOutputNeurons = 0;
    OutputNeuron* outputNeurons = nullptr;
};
static_assert(sizeof(OutputInterface) == 16);

//==================================================================================================

struct InputInterface {
    uint32_t targetBrickId = UNINIT_STATE_32;
    uint32_t numberOfInputNeurons = 0;
    InputNeuron* inputNeurons = nullptr;
};
static_assert(sizeof(InputInterface) == 16);

//==================================================================================================

struct SynapseConnection {
    SourceLocationPtr origin;
    float lowerBound = 0.0f;
    float potentialRange = std::numeric_limits<float>::max();

    SynapseConnection()
    {
        origin.brickId = UNINIT_STATE_16;
        origin.blockId = UNINIT_STATE_16;
        origin.neuronId = UNINIT_STATE_16;
        origin.posInNeuron = 0;
        origin.isInput = false;
    }
};
static_assert(sizeof(SynapseConnection) == 16);

//==================================================================================================

struct ConnectionBlock {
    SynapseConnection connections[NUMBER_OF_SYNAPSESECTION];
    uint64_t targetSynapseBlockPos = UNINIT_STATE_64;

    ConnectionBlock() { std::fill_n(connections, NUMBER_OF_SYNAPSESECTION, SynapseConnection()); }
};
static_assert(sizeof(ConnectionBlock) == 1032);

//==================================================================================================

struct Brick {
    uint32_t brickId = UNINIT_STATE_32;
    bool isOutputBrick = false;
    bool isInputBrick = false;
    bool wasResized = false;
    uint8_t padding1[1];

    Hanami::Position brickPos;
    uint32_t neighbors[12];
    uint32_t possibleBrickTargetIds[NUMBER_OF_POSSIBLE_NEXT];

    uint32_t dimX = 0;
    uint32_t dimY = 0;

    std::vector<ConnectionBlock> connectionBlocks;
    std::vector<NeuronBlock> neuronBlocks;
    std::vector<TempNeuronBlock> tempNeuronBlocks;

    Brick() { std::fill_n(neighbors, 12, UNINIT_STATE_32); }

    ~Brick() {}

    Brick(const Brick& other)
    {
        copyNormalPayload(other);

        for (const ConnectionBlock& block : other.connectionBlocks) {
            connectionBlocks.push_back(block);
        }

        for (const NeuronBlock& block : other.neuronBlocks) {
            neuronBlocks.push_back(block);
        }

        for (const TempNeuronBlock& block : other.tempNeuronBlocks) {
            tempNeuronBlocks.push_back(block);
        }
    }

    Brick& operator=(const Brick& other)
    {
        if (this != &other) {
            connectionBlocks.clear();
            neuronBlocks.clear();
            tempNeuronBlocks.clear();

            copyNormalPayload(other);

            for (const ConnectionBlock& block : other.connectionBlocks) {
                connectionBlocks.push_back(block);
            }

            for (const NeuronBlock& block : other.neuronBlocks) {
                neuronBlocks.push_back(block);
            }

            for (const TempNeuronBlock& block : other.tempNeuronBlocks) {
                tempNeuronBlocks.push_back(block);
            }
        }
        return *this;
    }

    Brick& operator=(Brick&& other)
    {
        if (this != &other) {
            connectionBlocks.clear();
            neuronBlocks.clear();
            tempNeuronBlocks.clear();

            copyNormalPayload(other);

            for (ConnectionBlock& block : other.connectionBlocks) {
                connectionBlocks.push_back(block);
            }

            for (NeuronBlock& block : other.neuronBlocks) {
                neuronBlocks.push_back(block);
            }

            for (TempNeuronBlock& block : other.tempNeuronBlocks) {
                tempNeuronBlocks.push_back(block);
            }
        }
        return *this;
    }

    void copyNormalPayload(const Brick& other)
    {
        brickId = other.brickId;
        isOutputBrick = other.isOutputBrick;
        isInputBrick = other.isInputBrick;
        wasResized = other.wasResized;

        brickPos = other.brickPos;
        memcpy(neighbors, other.neighbors, 12 * sizeof(uint32_t));
        memcpy(possibleBrickTargetIds,
               other.possibleBrickTargetIds,
               NUMBER_OF_POSSIBLE_NEXT * sizeof(uint32_t));

        dimX = other.dimX;
        dimY = other.dimY;
    }
};
static_assert(sizeof(Brick) == 512);

//==================================================================================================

struct CudaPointerHandle {
    uint32_t deviceId = 0;
    NeuronBlock* neuronBlocks = nullptr;
    TempNeuronBlock* tempNeuronBlock = nullptr;
    SynapseBlock* synapseBlocks = nullptr;
    std::vector<ConnectionBlock*> connectionBlocks;

    ClusterSettings* clusterSettings = nullptr;
};

//==================================================================================================

struct SourceLocation {
    Brick* brick = nullptr;
    NeuronBlock* neuronBlock = nullptr;
    TempNeuronBlock* tempNeuronBlock = nullptr;
    Neuron* neuron = nullptr;
    TempNeuron* tempNeuron = nullptr;
};

inline SourceLocation
getSourceNeuron(const SourceLocationPtr& location, Brick* bricks)
{
    SourceLocation sourceLoc;
    sourceLoc.brick = &bricks[location.brickId];

    sourceLoc.neuronBlock = &sourceLoc.brick->neuronBlocks[location.blockId];
    sourceLoc.tempNeuronBlock = &sourceLoc.brick->tempNeuronBlocks[location.blockId];

    sourceLoc.neuron = &sourceLoc.neuronBlock->neurons[location.neuronId];
    sourceLoc.tempNeuron = &sourceLoc.tempNeuronBlock->neurons[location.neuronId];

    return sourceLoc;
}

#endif  // HANAMI_CORE_SEGMENT_OBJECTS_H
