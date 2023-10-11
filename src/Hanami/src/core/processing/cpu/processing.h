/**
 * @file        processing.h
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

#ifndef HANAMI_CORE_PROCESSING_H
#define HANAMI_CORE_PROCESSING_H

#include <api/websocket/cluster_io.h>
#include <common.h>
#include <core/cluster/cluster.h>
#include <core/processing/cluster_io_functions.h>
#include <core/processing/objects.h>
#include <core/processing/section_update.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief get position of the highest output-position
 *
 * @param cluster output-cluster to check
 *
 * @return position of the highest output.
 */
inline uint32_t
getHighestOutput(const Cluster& cluster)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for (uint32_t outputNeuronId = 0; outputNeuronId < cluster.clusterHeader->outputValues.count;
         outputNeuronId++) {
        value = cluster.outputValues[outputNeuronId];
        if (value > hightest) {
            hightest = value;
            hightestPos = outputNeuronId;
        }
    }

    return hightestPos;
}

/**
 * @brief initialize a new specific synapse
 */
inline void
createNewSynapse(NeuronBlock* block,
                 Synapse* synapse,
                 const ClusterSettings* clusterSettings,
                 const float remainingW)
{
    const uint32_t* randomValues = HanamiRoot::m_randomValues;
    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = clusterSettings->signNeg;

    // set activation-border
    synapse->border = remainingW;

    // set target neuron
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId
        = static_cast<uint16_t>(randomValues[block->randomPos] % block->numberOfNeurons);

    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = (static_cast<float>(randomValues[block->randomPos]) / randMax) / 10.0f;

    // update weight with sign
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[block->randomPos] % 1000;
    synapse->weight *= static_cast<float>(1.0f - (1000.0f * sigNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

/**
 * @brief process synapse-section
 */
template <bool doTrain>
inline void
synapseProcessing(Cluster& cluster,
                  Synapse* section,
                  SynapseConnection* connection,
                  LocationPtr* currentLocation,
                  const float outH,
                  NeuronBlock* neuronBlocks,
                  SynapseBlock* synapseBlocks,
                  SynapseConnection* synapseConnections,
                  ClusterSettings* clusterSettings)
{
    uint32_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    // uint8_t active = 0;
    float counter = outH - connection->offset[currentLocation->sectionId];
    NeuronBlock* neuronBlock = &neuronBlocks[connection->targetNeuronBlockId];

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        synapse = &section[pos];

        if constexpr (doTrain) {
            // create new synapse if necesarry and training is active
            if (synapse->targetNeuronId == UNINIT_STATE_16) {
                createNewSynapse(neuronBlock, synapse, clusterSettings, counter);
            }

            if (synapse->border > 2.0f * counter && pos < SYNAPSES_PER_SYNAPSESECTION - 2) {
                const float val = synapse->border / 2.0f;
                section[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            // update target-neuron
            targetNeuron = &neuronBlock->neurons[synapse->targetNeuronId];
            if (counter >= synapse->border) {
                targetNeuron->input += synapse->weight;
            } else {
                targetNeuron->input += synapse->weight * ((1.0f / synapse->border) * counter);
            }
        }

        // update loop-counter
        counter -= synapse->border;
        ++pos;
    }

    LocationPtr* targetLocation = &connection->next[currentLocation->sectionId];

    if constexpr (doTrain) {
        if (counter > 0.01f && targetLocation->sectionId == UNINIT_STATE_16) {
            const float newOffset
                = (outH - counter) + connection->offset[currentLocation->sectionId];
            createNewSection(cluster, connection->origin[currentLocation->sectionId], newOffset);
            targetLocation = &connection->next[currentLocation->sectionId];
        }
    }

    if (targetLocation->blockId != UNINIT_STATE_32) {
        Synapse* nextSection
            = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
        SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
        synapseProcessing<doTrain>(cluster,
                                   nextSection,
                                   nextConnection,
                                   targetLocation,
                                   outH,
                                   neuronBlocks,
                                   synapseBlocks,
                                   synapseConnections,
                                   clusterSettings);
    }
}

/**
 * @brief process only a single neuron
 */
template <bool doTrain>
inline void
processSingleNeuron(Cluster& cluster,
                    Neuron* neuron,
                    const uint32_t blockId,
                    const uint32_t neuronIdInBlock,
                    NeuronBlock* neuronBlocks,
                    SynapseBlock* synapseBlocks,
                    SynapseConnection* synapseConnections,
                    ClusterSettings* clusterSettings)
{
    // handle active-state
    if (neuron->active == 0) {
        return;
    }

    LocationPtr* targetLocation = &neuron->target;

    if constexpr (doTrain) {
        if (targetLocation->blockId == UNINIT_STATE_32) {
            LocationPtr sourceLocation;
            sourceLocation.blockId = blockId;
            sourceLocation.sectionId = neuronIdInBlock;
            if (createNewSection(cluster, sourceLocation, 0.0f) == false) {
                return;
            }
            targetLocation = &neuron->target;
        }
    }

    if (targetLocation->blockId != UNINIT_STATE_32) {
        Synapse* nextSection
            = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
        SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
        synapseProcessing<doTrain>(cluster,
                                   nextSection,
                                   nextConnection,
                                   targetLocation,
                                   neuron->potential,
                                   neuronBlocks,
                                   synapseBlocks,
                                   synapseConnections,
                                   clusterSettings);
    }
}

/**
 * @brief process input brick
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrick(Cluster& cluster,
                           const Brick* brick,
                           float* inputValues,
                           NeuronBlock* neuronBlocks,
                           SynapseBlock* synapseBlocks,
                           SynapseConnection* synapseConnections,
                           ClusterSettings* clusterSettings)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->brickBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
         ++blockId) {
        block = &neuronBlocks[blockId];
        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < block->numberOfNeurons;
             ++neuronIdInBlock) {
            neuron = &block->neurons[neuronIdInBlock];
            neuron->potential = inputValues[counter];
            neuron->active = neuron->potential > 0.0f;

            processSingleNeuron<doTrain>(cluster,
                                         neuron,
                                         blockId,
                                         neuronIdInBlock,
                                         neuronBlocks,
                                         synapseBlocks,
                                         synapseConnections,
                                         clusterSettings);

            ++counter;
        }
    }
}

/**
 * @brief process normal internal brick
 */
template <bool doTrain>
inline void
processNeuronsOfNormalBrick(Cluster& cluster,
                            const Brick* brick,
                            NeuronBlock* neuronBlocks,
                            SynapseBlock* synapseBlocks,
                            SynapseConnection* synapseConnections,
                            ClusterSettings* clusterSettings)
{
    Neuron* neuron = nullptr;
    NeuronBlock* blocks = nullptr;

    // iterate over all neurons within the brick
    for (uint32_t blockId = brick->brickBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
         ++blockId) {
        blocks = &neuronBlocks[blockId];
        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < blocks->numberOfNeurons;
             ++neuronIdInBlock) {
            neuron = &blocks->neurons[neuronIdInBlock];

            neuron->potential /= clusterSettings->neuronCooldown;
            neuron->refractionTime = neuron->refractionTime >> 1;

            if (neuron->refractionTime == 0) {
                neuron->potential = clusterSettings->potentialOverflow * neuron->input;
                neuron->refractionTime = clusterSettings->refractionTime;
            }

            // update neuron
            neuron->potential -= neuron->border;
            neuron->active = neuron->potential > 0.0f;
            neuron->input = 0.0f;
            neuron->potential = log2(neuron->potential + 1.0f);

            processSingleNeuron<doTrain>(cluster,
                                         neuron,
                                         blockId,
                                         neuronIdInBlock,
                                         neuronBlocks,
                                         synapseBlocks,
                                         synapseConnections,
                                         clusterSettings);
        }
    }
}

/**
 * @brief process all neurons within a cluster
 */
inline void
prcessCoreSegment(Cluster& cluster)
{
    float* inputValues = cluster.inputValues;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseConnection* synapseConnections = cluster.synapseConnections;
    SynapseBlock* synapseBlocks = cluster.synapseBlocks;
    ClusterSettings* clusterSettings = cluster.clusterSettings;

    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for (uint32_t pos = 0; pos < numberOfBricks; ++pos) {
        const uint32_t brickId = cluster.brickOrder[pos];
        Brick* brick = &cluster.bricks[brickId];
        if (brick->isInputBrick) {
            if (clusterSettings->doTrain) {
                processNeuronsOfInputBrick<true>(cluster,
                                                 brick,
                                                 inputValues,
                                                 neuronBlocks,
                                                 synapseBlocks,
                                                 synapseConnections,
                                                 clusterSettings);
            } else {
                processNeuronsOfInputBrick<false>(cluster,
                                                  brick,
                                                  inputValues,
                                                  neuronBlocks,
                                                  synapseBlocks,
                                                  synapseConnections,
                                                  clusterSettings);
            }
        } else if (brick->isOutputBrick) {
            processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks);
        } else {
            if (clusterSettings->doTrain) {
                processNeuronsOfNormalBrick<true>(cluster,
                                                  brick,
                                                  neuronBlocks,
                                                  synapseBlocks,
                                                  synapseConnections,
                                                  clusterSettings);
            } else {
                processNeuronsOfNormalBrick<false>(cluster,
                                                   brick,
                                                   neuronBlocks,
                                                   synapseBlocks,
                                                   synapseConnections,
                                                   clusterSettings);
            }
        }
    }
}

#endif  // HANAMI_CORE_PROCESSING_H
