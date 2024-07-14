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

#include <core/cluster/cluster.h>
#include <core/cluster/objects.h>
#include <core/processing/cluster_resize.h>
#include <core/processing/logical_host.h>
#include <hanami_crypto/hashes.h>
#include <math.h>

#include <iostream>

/**
 * @brief processNeuronsOfInputHexagonBackward
 *
 * @param hexagon
 * @param inputValues
 * @param neuronBlocks
 */
template <bool doTrain>
inline void
processNeuronsOfInputHexagon(Cluster& cluster, InputInterface* inputInterface, Hexagon* hexagon)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;
    uint16_t blockId = 0;
    uint8_t neuronId = 0;

    // iterate over all neurons within the hexagon
    for (NeuronBlock& neuronBlock : hexagon->neuronBlocks) {
        for (neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
            if (counter >= inputInterface->inputNeurons.size()) {
                return;
            }
            neuron = &neuronBlock.neurons[neuronId];
            neuron->potential = inputInterface->inputNeurons[counter].value;
            neuron->active = neuron->potential > 0.0f;

            if constexpr (doTrain) {
                if (neuron->active != 0 && neuron->inUse == 0) {
                    SourceLocationPtr originLocation;
                    originLocation.hexagonId = hexagon->header.hexagonId;
                    originLocation.blockId = blockId;
                    originLocation.neuronId = neuronId;
                    createNewSection(cluster,
                                     originLocation,
                                     0.0f,
                                     std::numeric_limits<float>::max(),
                                     cluster.attachedHost->synapseBlocks);
                }
            }
            counter++;
        }
        blockId++;
    }
}

// Derivative of the activation function
inline float
sigmoidDerivative(const float x)
{
    return x * (1 - x);
}

/**
 * @brief processNeuronsOfOutputHexagon
 * @param hexagons
 * @param outputInterface
 * @param neuronBlocks
 * @param hexagonId
 * @param randomSeed
 */
template <bool doTrain>
inline void
processNeuronsOfOutputHexagon(std::vector<Hexagon>& hexagons,
                              OutputInterface* outputInterface,
                              const uint32_t hexagonId,
                              uint32_t randomSeed)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    Hexagon* hexagon = nullptr;
    OutputNeuron* out = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    uint32_t neuronBlockId = 0;
    float weightSum = 0.0f;
    bool found = false;

    hexagon = &hexagons[hexagonId];
    for (NeuronBlock& block : hexagon->neuronBlocks) {
        for (uint64_t j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &block.neurons[j];
            neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->input));
            neuron->input = 0.0f;
        }
    }

    for (uint64_t i = 0; i < outputInterface->outputNeurons.size(); ++i) {
        out = &outputInterface->outputNeurons[i];
        hexagon = &hexagons[outputInterface->targetHexagonId];
        weightSum = 0.0f;

        for (uint8_t j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if constexpr (doTrain) {
                found = false;
                randomSeed = Hanami::pcg_hash(randomSeed);
                if (found == false && target->blockId == UNINIT_STATE_16 && out->exprectedVal > 0.0
                    && randomSeed % 50 == 0)
                {
                    randomSeed = Hanami::pcg_hash(randomSeed);
                    const uint32_t blockId = randomSeed % hexagon->neuronBlocks.size();
                    randomSeed = Hanami::pcg_hash(randomSeed);
                    const uint16_t neuronId = randomSeed % NEURONS_PER_NEURONBLOCK;
                    const float potential
                        = hexagon->neuronBlocks[blockId].neurons[neuronId].potential;

                    if (potential != 0.5f) {
                        target->blockId = blockId;
                        target->neuronId = neuronId;
                        randomSeed = Hanami::pcg_hash(randomSeed);
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

            neuron = &hexagon->neuronBlocks[target->blockId].neurons[target->neuronId];
            weightSum += neuron->potential * target->connectionWeight;
        }

        out->outputVal = 0.0f;
        if (weightSum != 0.0f) {
            out->outputVal = 1.0f / (1.0f + exp(-1.0f * weightSum));
        }
        // std::cout << out->outputVal << " : " << out->exprectedVal << std::endl;
    }
    // std::cout << "-------------------------------------" << std::endl;
}

/**
 * @brief backpropagateOutput
 *
 * @param hexagon
 * @param neuronBlocks
 * @param tempNeuronBlocks
 * @param outputValues
 * @param expectedValues
 * @param settings
 *
 * @return
 */
inline bool
backpropagateOutput(std::vector<Hexagon>& hexagons,
                    OutputInterface* outputInterface,
                    const ClusterSettings* settings,
                    const uint32_t hexagonId)
{
    Neuron* neuron = nullptr;
    Hexagon* hexagon = nullptr;
    OutputNeuron* out = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    float totalDelta = 0.0f;
    float learnValue = 0.1f;
    float delta = 0.0f;
    float update = 0.0f;
    uint64_t i = 0;
    uint64_t j = 0;

    for (i = 0; i < outputInterface->outputNeurons.size(); ++i) {
        out = &outputInterface->outputNeurons[i];
        hexagon = &hexagons[outputInterface->targetHexagonId];
        delta = out->outputVal - out->exprectedVal;
        update = delta * sigmoidDerivative(out->outputVal);

        for (j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out->targets[j];

            if (target->blockId == UNINIT_STATE_16) {
                continue;
            }

            neuron = &hexagon->neuronBlocks[target->blockId].neurons[target->neuronId];
            neuron->delta += update * target->connectionWeight;
            target->connectionWeight -= update * learnValue * neuron->potential;

            totalDelta += abs(delta);
        }
    }

    hexagon = &hexagons[hexagonId];
    for (i = 0; i < hexagon->neuronBlocks.size(); ++i) {
        for (j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &hexagon->neuronBlocks[i].neurons[j];
            neuron->delta *= sigmoidDerivative(neuron->potential);
        }
    }

    return true;
    // return totalDelta > settings->backpropagationBorder;
}

#endif  // HANAMI_CORE_CLUSTER_IO_FUNCTIONS_H
