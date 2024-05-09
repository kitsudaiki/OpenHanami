/**
 * @file        bricks.h
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

#ifndef BRICKS_H
#define BRICKS_H

#include <core/processing/objects.h>

struct BrickHeader {
   public:
    uint32_t brickId = UNINIT_STATE_32;
    bool isInputBrick = false;
    bool isOutputBrick = false;
    uint8_t padding1[2];
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    Hanami::Position brickPos;
};
static_assert(sizeof(BrickHeader) == 32);

//==================================================================================================

class Brick
{
   public:
    BrickHeader header;

    InputInterface* inputInterface = nullptr;
    OutputInterface* outputInterface = nullptr;

    std::vector<ConnectionBlock> connectionBlocks;
    std::vector<NeuronBlock> neuronBlocks;
    std::vector<TempNeuronBlock> tempNeuronBlocks;

    bool wasResized = false;
    uint32_t possibleBrickTargetIds[NUMBER_OF_POSSIBLE_NEXT];
    uint32_t neighbors[12];

    Brick();
    ~Brick();

    Brick(const Brick& other);

    Brick& operator=(const Brick& other);
    Brick& operator=(Brick&& other);
};

//==================================================================================================

struct SourceLocation {
    Brick* brick = nullptr;
    NeuronBlock* neuronBlock = nullptr;
    TempNeuronBlock* tempNeuronBlock = nullptr;
    Neuron* neuron = nullptr;
    TempNeuron* tempNeuron = nullptr;
};

/**
 * @brief getSourceNeuron
 * @param location
 * @param bricks
 * @return
 */
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

#endif  // BRICKS_H
