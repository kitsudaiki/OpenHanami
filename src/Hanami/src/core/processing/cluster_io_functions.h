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
                                   std::vector<float>& inputValues,
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

// Derivative of the activation function
inline float
sigmoidDerivative(const float x)
{
    return x * (1 - x);
}

/**
 * @brief processNeuronsOfOutputBrick
 *
 * @param brick
 * @param outputValues
 * @param neuronBlocks
 */
template <bool doTrain>
inline void
processNeuronsOfOutputBrick(std::vector<Brick>& bricks,
                            std::vector<OutputNeuron>& outputNeurons,
                            std::vector<NeuronBlock>& neuronBlocks,
                            std::vector<float>& outputValues,
                            const std::vector<float>& expectedValues,
                            const uint32_t brickId)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    const uint64_t numberOfOutputs = outputNeurons.size();
    Brick* brick = nullptr;
    OutputNeuron* out = nullptr;
    TargetLocationPtr* target = nullptr;
    float weightSum = 0.0f;
    bool found = false;

    for (uint64_t i = 0; i < numberOfOutputs; ++i) {
        brick = &bricks[brickId];
        const uint32_t neuronBlockId = brick->neuronBlockPos + 0;
        neuron = &neuronBlocks[neuronBlockId].neurons[i];
        neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->input));
        neuron->input = 0.0f;
    }

    for (uint64_t i = 0; i < 10; ++i) {
        out = &outputNeurons[i];
        brick = &bricks[out->brickId];
        weightSum = 0.0f;

        for (uint8_t j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if constexpr (doTrain) {
                found = false;
                if (found == false && target->blockId == UNINIT_STATE_32 && expectedValues[i] > 0.0
                    && rand() % 10 == 0)
                {
                    const uint32_t blockId = 0;
                    const uint16_t neuronId = rand() % 30;

                    const uint32_t totalBlockId = brick->neuronBlockPos + blockId;
                    const float potential = neuronBlocks[totalBlockId].neurons[neuronId].potential;

                    if (potential != 0.5f) {
                        target->blockId = blockId;
                        target->neuronId = neuronId;
                        target->weight = ((float)rand() / (float)RAND_MAX);
                        found = true;
                    }
                }
            }

            if (target->blockId == UNINIT_STATE_32) {
                continue;
            }

            const uint32_t neuronBlockId = brick->neuronBlockPos + target->blockId;
            neuron = &neuronBlocks[neuronBlockId].neurons[target->neuronId];
            weightSum += neuron->potential * target->weight;
        }

        outputValues[i] = 0.0f;
        if (weightSum != 0.0f) {
            outputValues[i] = 1.0f / (1.0f + exp(-1.0f * weightSum));
        }
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
backpropagateOutput(std::vector<Brick>& bricks,
                    std::vector<OutputNeuron>& outputNeurons,
                    std::vector<NeuronBlock>& neuronBlocks,
                    std::vector<TempNeuronBlock>& tempNeuronBlocks,
                    const std::vector<float>& outputValues,
                    const std::vector<float>& expectedValues,
                    const ClusterSettings* settings,
                    const uint32_t brickId)
{
    Neuron* neuron = nullptr;
    TempNeuron* tempNeuron = nullptr;
    const uint64_t numberOfOutputs = outputNeurons.size();
    Brick* brick = nullptr;
    OutputNeuron* out = nullptr;
    float totalDelta = 0.0f;
    float learnValue = 0.0f;
    TargetLocationPtr* target = nullptr;

    for (uint64_t i = 0; i < 10; ++i) {
        out = &outputNeurons[i];
        brick = &bricks[out->brickId];
        const float delta = outputValues[i] - expectedValues[i];
        const float update = delta * sigmoidDerivative(outputValues[i]);

        learnValue = abs(delta) + 0.1f;

        for (uint8_t j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if (target->blockId == UNINIT_STATE_32) {
                continue;
            }

            const uint32_t neuronBlockId = brick->neuronBlockPos + target->blockId;

            tempNeuron = &tempNeuronBlocks[neuronBlockId].neurons[target->neuronId];
            neuron = &neuronBlocks[neuronBlockId].neurons[target->neuronId];

            tempNeuron->delta[0] += update * target->weight;
            target->weight -= update * learnValue * neuron->potential;

            totalDelta += abs(delta);
        }
    }

    for (uint64_t i = 0; i < numberOfOutputs; ++i) {
        brick = &bricks[brickId];
        const uint32_t neuronBlockId = brick->neuronBlockPos + 0;
        tempNeuron = &tempNeuronBlocks[neuronBlockId].neurons[i];
        neuron = &neuronBlocks[neuronBlockId].neurons[i];

        tempNeuron->delta[0] *= sigmoidDerivative(neuron->potential);
    }

    // std::cout << "totalDelta: " << totalDelta << std::endl;
    // return totalDelta > settings->backpropagationBorder;
    return true;
}

#endif  // HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
