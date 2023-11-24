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
 * @brief processNeuronsOfInputBrick
 * @param cluster
 * @param brick
 * @param inputValues
 * @param neuronBlocks
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrick(Cluster& cluster,
                           const Brick* brick,
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
         blockId++)
    {
        block = &neuronBlocks[blockId];
        for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < block->numberOfNeurons;
             neuronIdInBlock++)
        {
            neuron = &block->neurons[neuronIdInBlock];
            neuron->potential = brickBuffer[counter];
            neuron->active = neuron->potential > 0.0f;
            if constexpr (doTrain) {
                if (neuron->active != 0 && neuron->inUse == 0) {
                    SourceLocationPtr originLocation;
                    originLocation.blockId = blockId;
                    originLocation.sectionId = neuronIdInBlock;
                    neuron->inUse = createNewSection(cluster, originLocation, 0.0f, 0);
                }
            }
            counter++;
        }
    }
}

/**
 * @brief createNewSynapse
 * @param block
 * @param synapse
 * @param clusterSettings
 * @param remainingW
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
}

/**
 * @brief synapseProcessingBackward
 * @param cluster
 * @param section
 * @param connection
 * @param targetNeuronBlock
 * @param sourceNeuron
 * @param originLocation
 * @param clusterSettings
 */
template <bool doTrain>
void
synapseProcessingBackward(Cluster& cluster,
                          Synapse* section,
                          SynapseConnection* connection,
                          NeuronBlock* targetNeuronBlock,
                          Neuron* sourceNeuron,
                          const SourceLocationPtr originLocation,
                          ClusterSettings* clusterSettings)
{
    float counter = sourceNeuron->potential - connection->offset;
    uint pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        synapse = &section[pos];

        if constexpr (doTrain) {
            // create new synapse if necesarry and training is active
            if (synapse->targetNeuronId == UNINIT_STATE_16) {
                createNewSynapse(targetNeuronBlock, synapse, clusterSettings, counter);
            }

            if (synapse->border > 2.0f * counter && pos < SYNAPSES_PER_SYNAPSESECTION - 2) {
                const float val = synapse->border / 2.0f;
                section[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            // update target-neuron
            targetNeuron = &targetNeuronBlock->neurons[synapse->targetNeuronId];
            if (counter >= synapse->border) {
                targetNeuron->input += synapse->weight;
            }
            else {
                targetNeuron->input += synapse->weight * ((1.0f / synapse->border) * counter);
            }
        }

        // update loop-counter
        counter -= synapse->border;
        ++pos;
    }

    if constexpr (doTrain) {
        if (counter > 0.01f && connection->hasNext == false) {
            const float newOffset = (sourceNeuron->potential - counter) + connection->offset;
            connection->hasNext = createNewSection(
                cluster, originLocation, newOffset, connection->origin.posInNeuron + 1);
        }
    }
}

/**
 * @brief processBrick
 * @param cluster
 * @param brick
 * @param neuronBlocks
 * @param synapseBlocks
 * @param clusterSettings
 */
template <bool doTrain>
inline void
processBrick(Cluster& cluster,
             Brick* brick,
             NeuronBlock* neuronBlocks,
             SynapseBlock* synapseBlocks,
             ClusterSettings* clusterSettings)
{
    SynapseConnection* scon = nullptr;
    Neuron* neuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    Synapse* section = nullptr;
    uint32_t counter = 0;

    // process synapses
    for (ConnectionBlock& connection : brick->connectionBlocks) {
        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            scon = &connection.connections[i];
            if (scon->origin.blockId == UNINIT_STATE_32) {
                continue;
            }
            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.sectionId];
            if (sourceNeuron->active == 0) {
                continue;
            }

            section = synapseBlocks[connection.targetSynapseBlockPos].synapses[i];
            targetNeuronBlock = &neuronBlocks[brick->neuronBlockPos + (counter / brick->dimY)];

            synapseProcessingBackward<doTrain>(cluster,
                                               section,
                                               scon,
                                               targetNeuronBlock,
                                               sourceNeuron,
                                               scon->origin,
                                               clusterSettings);
        }

        ++counter;
    }

    // process neurons
    if (brick->isOutputBrick == false) {
        for (uint32_t blockId = brick->neuronBlockPos;
             blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
             ++blockId)
        {
            targetNeuronBlock = &neuronBlocks[blockId];
            for (uint32_t neuronIdInBlock = 0; neuronIdInBlock < NEURONS_PER_NEURONSECTION;
                 ++neuronIdInBlock)
            {
                neuron = &targetNeuronBlock->neurons[neuronIdInBlock];
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

                if constexpr (doTrain) {
                    if (neuron->active != 0 && neuron->inUse == 0) {
                        SourceLocationPtr originLocation;
                        originLocation.blockId = blockId;
                        originLocation.sectionId = neuronIdInBlock;
                        neuron->inUse = createNewSection(cluster, originLocation, 0.0f, 0);
                    }
                }
            }
        }
    }
}

/**
 * @brief prcessCoreSegment
 * @param cluster
 * @param doTrain
 */
inline void
prcessCoreSegment(Cluster& cluster, const bool doTrain)
{
    float* inputValues = cluster.inputValues;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::m_synapseBlocks);
    ClusterSettings* clusterSettings = &cluster.clusterHeader->settings;

    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        Brick* brick = &cluster.bricks[brickId];
        if (brick->isInputBrick) {
            if (doTrain) {
                processNeuronsOfInputBrick<true>(cluster, brick, inputValues, neuronBlocks);
            }
            else {
                processNeuronsOfInputBrick<false>(cluster, brick, inputValues, neuronBlocks);
            }
            continue;
        }

        if (doTrain) {
            processBrick<true>(cluster, brick, neuronBlocks, synapseBlocks, clusterSettings);
        }
        else {
            processBrick<false>(cluster, brick, neuronBlocks, synapseBlocks, clusterSettings);
        }

        if (brick->isOutputBrick) {
            processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks);
        }
    }
}

#endif  // HANAMI_CORE_PROCESSING_H
