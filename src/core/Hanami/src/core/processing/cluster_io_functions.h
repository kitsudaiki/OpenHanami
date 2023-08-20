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

#include <iostream>
#include <math.h>
#include "objects.h"

/**
 * @brief process input brick
 */
inline void
processNeuronsOfInputBrickBackward(const Brick* brick,
                                   float* inputValues,
                                   NeuronBlock* neuronBlocks)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
        blockId++)
    {
        block = &neuronBlocks[blockId];
        for(uint32_t neuronIdInBlock = 0;
            neuronIdInBlock < block->numberOfNeurons;
            neuronIdInBlock++)
        {
            neuron = &block->neurons[neuronIdInBlock];
            neuron->potential = inputValues[counter];
            neuron->active = neuron->potential > 0.0f;
            neuron->isNew = neuron->active != 0 && neuron->target.blockId == UNINIT_STATE_32;
            neuron->newOffset = 0.0f;
            counter++;
        }
    }
}

inline void
processNeuronsOfOutputBrick(const Brick* brick,
                            float* outputValues,
                            NeuronBlock* neuronBlocks)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
        blockId++)
    {
        block = &neuronBlocks[blockId];
        for(uint32_t neuronIdInBlock = 0;
            neuronIdInBlock < block->numberOfNeurons;
            neuronIdInBlock++)
        {
            neuron = &block->neurons[neuronIdInBlock];
            neuron->potential = neuron->input;
            if(neuron->potential != 0.0f) {
                neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->potential));
            }
            //std::cout<<"neuron->potential: "<<neuron->potential<<std::endl;
            outputValues[counter] = neuron->potential;
            neuron->input = 0.0f;
            counter++;
        }
    }
}

/**
 * @brief backpropagate values of an output-brick
 */
inline bool
backpropagateOutput(const Brick* brick,
                    NeuronBlock* neuronBlocks,
                    float* outputValues,
                    float* expectedValues,
                    SegmentSettings* settings)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    float totalDelta = 0.0f;
    uint32_t counter = 0;

    // iterate over all neurons within the brick
    for(uint32_t neuronSectionId = brick->brickBlockPos;
        neuronSectionId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
        neuronSectionId++)
    {
        block = &neuronBlocks[neuronSectionId];
        for(uint32_t neuronIdInBlock = 0;
            neuronIdInBlock < block->numberOfNeurons;
            neuronIdInBlock++)
        {
            neuron = &block->neurons[neuronIdInBlock];
            neuron->delta = outputValues[counter] - expectedValues[counter];
            neuron->delta *= outputValues[counter] * (1.0f - outputValues[counter]);
            //std::cout<<" expectedValues[counter] : "<< expectedValues[counter] <<"    outputValues[counter] : "<< outputValues[counter] <<std::endl;
            totalDelta += abs(neuron->delta);
            counter++;
        }
    }
    return totalDelta > settings->backpropagationBorder;
}

#endif // HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
