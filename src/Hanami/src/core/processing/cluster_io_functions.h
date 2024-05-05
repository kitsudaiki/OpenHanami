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
 * @brief function for generating random-values
 *        coming from this website:
 *            https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 *
 * @param input seed for random value
 *
 * @return random value
 */
inline uint32_t
pcg_hash2(const uint32_t input)
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * @brief processNeuronsOfInputBrickBackward
 *
 * @param brick
 * @param inputValues
 * @param neuronBlocks
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrickBackward(Brick* brick,
                                   InputInterface& inputInterface,
                                   NeuronBlock* neuronBlocks)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;

    // iterate over all neurons within the brick
    for (NeuronBlock& block : brick->neuronBlocks) {
        for (uint32_t neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
            neuron = &block.neurons[neuronId];
            neuron->potential = inputInterface.inputNeurons[counter].value;
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
 * @param bricks
 * @param outputInterface
 * @param neuronBlocks
 * @param brickId
 * @param randomSeed
 */
template <bool doTrain>
inline void
processNeuronsOfOutputBrick(std::vector<Brick>& bricks,
                            OutputInterface* outputInterface,
                            const uint32_t brickId,
                            uint32_t randomSeed)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    Brick* brick = nullptr;
    OutputNeuron* out = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    uint32_t neuronBlockId = 0;
    float weightSum = 0.0f;
    bool found = false;

    brick = &bricks[brickId];
    for (NeuronBlock& block : brick->neuronBlocks) {
        for (uint64_t j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &block.neurons[j];
            neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->input));
            neuron->input = 0.0f;
        }
    }

    for (uint64_t i = 0; i < outputInterface->numberOfOutputNeurons; ++i) {
        out = &outputInterface->outputNeurons[i];
        brick = &bricks[outputInterface->targetBrickId];
        weightSum = 0.0f;

        for (uint8_t j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if constexpr (doTrain) {
                found = false;
                randomSeed = pcg_hash2(randomSeed);
                if (found == false && target->blockId == UNINIT_STATE_16 && out->exprectedVal > 0.0
                    && randomSeed % 50 == 0)
                {
                    randomSeed = pcg_hash2(randomSeed);
                    const uint32_t blockId = randomSeed % brick->neuronBlocks.size();
                    randomSeed = pcg_hash2(randomSeed);
                    const uint16_t neuronId = randomSeed % NEURONS_PER_NEURONBLOCK;
                    const float potential
                        = brick->neuronBlocks[blockId].neurons[neuronId].potential;

                    if (potential != 0.5f) {
                        target->blockId = blockId;
                        target->neuronId = neuronId;
                        randomSeed = pcg_hash2(randomSeed);
                        target->connectionWeight = ((float)randomSeed / (float)RAND_MAX);
                        found = true;

                        if (potential < 0.5f) {
                            target->connectionWeight *= -1.0f;
                        }
                    }
                }
            }

            if (target->blockId == UNINIT_STATE_16) {
                continue;
            }

            neuron = &brick->neuronBlocks[target->blockId].neurons[target->neuronId];
            weightSum += neuron->potential * target->connectionWeight;
        }

        out->outputVal = 0.0f;
        if (weightSum != 0.0f) {
            out->outputVal = 1.0f / (1.0f + exp(-1.0f * weightSum));
        }
        // std::cout<<out->outputVal<<" : "<<out->exprectedVal<<std::endl;
    }
    //   std::cout<<"-------------------------------------"<<std::endl;
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
                    OutputInterface* outputInterface,
                    const ClusterSettings* settings,
                    const uint32_t brickId)
{
    Neuron* neuron = nullptr;
    Brick* brick = nullptr;
    OutputNeuron* out = nullptr;
    TempNeuron* tempNeuron = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    float totalDelta = 0.0f;
    float learnValue = 0.1f;
    float delta = 0.0f;
    float update = 0.0f;
    uint64_t i = 0;
    uint64_t j = 0;

    for (i = 0; i < outputInterface->numberOfOutputNeurons; ++i) {
        out = &outputInterface->outputNeurons[i];
        brick = &bricks[outputInterface->targetBrickId];
        delta = out->outputVal - out->exprectedVal;
        update = delta * sigmoidDerivative(out->outputVal);

        for (j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if (target->blockId == UNINIT_STATE_16) {
                continue;
            }

            tempNeuron = &brick->tempNeuronBlocks[target->blockId].neurons[target->neuronId];
            neuron = &brick->neuronBlocks[target->blockId].neurons[target->neuronId];

            tempNeuron->delta[0] += update * target->connectionWeight;
            target->connectionWeight -= update * learnValue * neuron->potential;

            totalDelta += abs(delta);
        }
    }

    brick = &bricks[brickId];
    for (i = 0; i < brick->neuronBlocks.size(); ++i) {
        for (j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &brick->neuronBlocks[i].neurons[j];
            tempNeuron = &brick->tempNeuronBlocks[i].neurons[j];
            tempNeuron->delta[0] *= sigmoidDerivative(neuron->potential);
        }
    }

    return true;
    // return totalDelta > settings->backpropagationBorder;
}

#endif  // HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
