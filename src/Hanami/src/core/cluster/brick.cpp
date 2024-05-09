/**
 * @file        bricks.cpp
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

#include "brick.h"

/**
 * @brief BrickData::BrickData
 * @param other
 */
Brick::Brick(const Brick& other)
{
    header = other.header;

    inputInterface = other.inputInterface;
    outputInterface = other.outputInterface;
    memcpy(neighbors, other.neighbors, 12 * sizeof(uint32_t));
    memcpy(possibleBrickTargetIds,
           other.possibleBrickTargetIds,
           NUMBER_OF_POSSIBLE_NEXT * sizeof(uint32_t));

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

/**
 * @brief BrickData::BrickData
 */
Brick::Brick() { std::fill_n(neighbors, 12, UNINIT_STATE_32); }

/**
 * @brief BrickData::~BrickData
 */
Brick::~Brick() {}

/**
 * @brief BrickData::operator =
 * @param other
 * @return
 */
Brick&
Brick::operator=(const Brick& other)
{
    if (this != &other) {
        connectionBlocks.clear();
        neuronBlocks.clear();
        tempNeuronBlocks.clear();

        header = other.header;

        inputInterface = other.inputInterface;
        outputInterface = other.outputInterface;
        memcpy(neighbors, other.neighbors, 12 * sizeof(uint32_t));
        memcpy(possibleBrickTargetIds,
               other.possibleBrickTargetIds,
               NUMBER_OF_POSSIBLE_NEXT * sizeof(uint32_t));

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

/**
 * @brief BrickData::operator =
 * @param other
 * @return
 */
Brick&
Brick::operator=(Brick&& other)
{
    if (this != &other) {
        connectionBlocks.clear();
        neuronBlocks.clear();
        tempNeuronBlocks.clear();

        header = other.header;

        inputInterface = other.inputInterface;
        outputInterface = other.outputInterface;
        memcpy(neighbors, other.neighbors, 12 * sizeof(uint32_t));
        memcpy(possibleBrickTargetIds,
               other.possibleBrickTargetIds,
               NUMBER_OF_POSSIBLE_NEXT * sizeof(uint32_t));

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
