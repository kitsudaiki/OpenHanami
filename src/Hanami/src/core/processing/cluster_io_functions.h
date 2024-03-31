/**
 * @file        cluster_io_functions.h
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

#ifndef HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
#define HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H

#include <math.h>

#include <iostream>

#include "objects.h"

/**
 * @brief processNeuronsOfInputBrickBackward
 *
 * @param brick
 * @param inputValues
 * @param neuronBlocks
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrickBackward(const Brick* brick,
                                   float* inputValues,
                                   NeuronBlock* neuronBlocks)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;
    float* brickBuffer = &inputValues[brick->ioBufferPos];

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->neuronBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         ++blockId)
    {
        block = &neuronBlocks[blockId];
        for (uint32_t neuronId = 0; neuronId < block->numberOfNeurons; ++neuronId) {
            neuron = &block->neurons[neuronId];
            neuron->potential = brickBuffer[counter];
            neuron->active = neuron->potential > 0.0f;
            if constexpr (doTrain) {
                neuron->isNew = neuron->active != 0 && neuron->inUse == 0;
                neuron->newLowerBound = 0.0f;
            }
            counter++;
        }
    }
}

/**
 * @brief processNeuronsOfOutputBrick
 *
 * @param brick
 * @param outputValues
 * @param neuronBlocks
 */
inline void
processNeuronsOfOutputBrick(const Brick* brick,
                            float* outputValues,
                            NeuronBlock* neuronBlocks,
                            const uint32_t blockId)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;
    float* brickBuffer = &outputValues[brick->ioBufferPos];
    const uint32_t neuronBlockId = brick->neuronBlockPos + blockId;

    block = &neuronBlocks[neuronBlockId];
    for (uint32_t neuronId = 0; neuronId < block->numberOfNeurons; ++neuronId) {
        neuron = &block->neurons[neuronId];
        neuron->potential = neuron->input;
        if (neuron->potential != 0.0f) {
            neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->potential));
        }
        brickBuffer[counter + (blockId * NEURONS_PER_NEURONSECTION)] = neuron->potential;
        neuron->input = 0.0f;
        counter++;
    }
}

/**
 * @brief backpropagateOutput
 *
 * @param brick
 * @param neuronBlocks
 * @param tempNeuronBlocks
 * @param outputValues
 * @param expectedValues
 * @param settings
 *
 * @return
 */
inline bool
backpropagateOutput(const Brick* brick,
                    NeuronBlock* neuronBlocks,
                    TempNeuronBlock* tempNeuronBlocks,
                    float* outputValues,
                    float* expectedValues,
                    ClusterSettings* settings)
{
    NeuronBlock* block = nullptr;
    TempNeuron* tempNeuron = nullptr;
    TempNeuronBlock* tempBlock = nullptr;
    float totalDelta = 0.0f;
    uint32_t counter = 0;
    float* outputBuffer = &outputValues[brick->ioBufferPos];
    float* expectedBuffer = &expectedValues[brick->ioBufferPos];

    // iterate over all neurons within the brick
    for (uint32_t neuronSectionId = brick->neuronBlockPos;
         neuronSectionId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         neuronSectionId++)
    {
        block = &neuronBlocks[neuronSectionId];
        tempBlock = &tempNeuronBlocks[neuronSectionId];

        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < block->numberOfNeurons;
             neuronIdInBlock++)
        {
            tempNeuron = &tempBlock->neurons[neuronIdInBlock];
            tempNeuron->delta[0] = outputBuffer[counter] - expectedBuffer[counter];
            tempNeuron->delta[0] *= outputBuffer[counter] * (1.0f - outputBuffer[counter]);
            totalDelta += abs(tempNeuron->delta[0]);
            counter++;
        }
    }

    return totalDelta > settings->backpropagationBorder;
}

#endif  // HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
