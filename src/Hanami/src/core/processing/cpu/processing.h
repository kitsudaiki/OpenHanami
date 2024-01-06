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
 * @brief process all neurons of an input-brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick input-brick to process
 * @param inputValues pointer to buffer with all input-values to apply
 * @param neuronBlocks pointer to the buffer with the neuron-blocks
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrick(Cluster& cluster, const Brick* brick)
{
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    float* inputValues = cluster.inputValues;
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
                    originLocation.neuronId = neuronIdInBlock;
                    createNewSection(cluster, originLocation, 0.0f, HanamiRoot::cpuSynapseBlocks);
                }
            }
            counter++;
        }
    }
}

/**
 * @brief initialize a new synpase
 *
 * @param block source-neuron-block, which is only used to hold the randamo-value
 * @param synapse pointer to the synapse, which should be (re-) initialized
 * @param clusterSettings pointer to the cluster-settings
 * @param remainingW new weight for the synapse
 */
inline void
createNewSynapse(NeuronBlock* block,
                 Synapse* synapse,
                 const ClusterSettings* clusterSettings,
                 const float remainingW,
                 uint32_t& randomSeed)
{
    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = clusterSettings->signNeg;

    // set activation-border
    synapse->border = remainingW;

    // set initial active-counter for reduction-process
    synapse->activeCounter = 5;

    // set target neuron
    randomSeed = pcg_hash(randomSeed);
    synapse->targetNeuronId = static_cast<uint16_t>(randomSeed % block->numberOfNeurons);

    randomSeed = pcg_hash(randomSeed);
    synapse->weight = (static_cast<float>(randomSeed) / randMax) / 10.0f;

    // update weight with sign
    randomSeed = pcg_hash(randomSeed);
    signRand = randomSeed % 1000;
    synapse->weight *= static_cast<float>(1.0f - (1000.0f * sigNeg > signRand) * 2);
}

/**
 * @brief process a single synapse-section
 *
 * @param cluster cluster, where the synapseSection belongs to
 * @param synapseSection current synapse-section to process
 * @param connection pointer to the connection-object, which is related to the section
 * @param targetNeuronBlock neuron-block, which is the target for all synapses in the section
 * @param sourceNeuron pointer to source-neuron, which had triggered the section
 * @param originLocation location of the source-neuron to mark updates
 * @param clusterSettings pointer to cluster-settings
 */
template <bool doTrain>
inline void
synapseProcessingBackward(Cluster& cluster,
                          SynapseSection* synapseSection,
                          SynapseConnection* connection,
                          NeuronBlock* targetNeuronBlock,
                          Neuron* sourceNeuron,
                          const SourceLocationPtr originLocation,
                          ClusterSettings* clusterSettings,
                          uint32_t& randomSeed)
{
    float potential = sourceNeuron->potential - connection->offset;
    float val = 0.0f;
    uint8_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && potential > 0.01f) {
        synapse = &synapseSection->synapses[pos];

        if constexpr (doTrain) {
            // create new synapse if necesarry and training is active
            if (synapse->targetNeuronId == UNINIT_STATE_8) {
                createNewSynapse(
                    targetNeuronBlock, synapse, clusterSettings, potential, randomSeed);
            }

            if (synapse->border > 2.0f * potential && pos < SYNAPSES_PER_SYNAPSESECTION - 2) {
                const float val = synapse->border / 2.0f;
                synapseSection->synapses[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            // update target-neuron
            targetNeuron = &targetNeuronBlock->neurons[synapse->targetNeuronId];
            val = synapse->weight;
            if (potential < synapse->border) {
                val *= ((1.0f / synapse->border) * potential);
            }
            targetNeuron->input += val;
        }

        // update loop-counter
        potential -= synapse->border;
        ++pos;
    }

    if constexpr (doTrain) {
        if (potential > 0.01f && synapseSection->hasNext == false) {
            const float newOffset = (sourceNeuron->potential - potential) + connection->offset;
            synapseSection->hasNext = createNewSection(
                cluster, originLocation, newOffset, HanamiRoot::cpuSynapseBlocks);
        }
    }
}

/**
 * @brief process all synapes of a brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick pointer to current brick
 */
template <bool doTrain>
inline void
processSynapses(Cluster& cluster, Brick* brick)
{
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(HanamiRoot::cpuSynapseBlocks);
    ClusterSettings* clusterSettings = &cluster.clusterHeader->settings;
    SynapseConnection* scon = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    SynapseSection* section = nullptr;
    uint32_t randomeSeed = rand();

    // process synapses
    for (uint32_t c = 0; c < brick->connectionBlocks.size(); c++) {
        connectionBlock = &brick->connectionBlocks[c];

        for (uint16_t i = 0; i < NUMBER_OF_SYNAPSESECTION; i++) {
            scon = &connectionBlock->connections[i];
            if (scon->origin.blockId == UNINIT_STATE_32) {
                continue;
            }

            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.neuronId];
            if (sourceNeuron->active == 0) {
                continue;
            }

            section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];
            targetNeuronBlock = &neuronBlocks[brick->neuronBlockPos + (c / brick->dimY)];
            randomeSeed += (c * NUMBER_OF_SYNAPSESECTION) + i;

            synapseProcessingBackward<doTrain>(cluster,
                                               section,
                                               scon,
                                               targetNeuronBlock,
                                               sourceNeuron,
                                               scon->origin,
                                               clusterSettings,
                                               randomeSeed);
        }
    }
}

/**
 * @brief process all neurons of a brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick pointer to current brick
 */
template <bool doTrain>
inline void
processNeurons(Cluster& cluster, Brick* brick)
{
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    ClusterSettings* clusterSettings = &cluster.clusterHeader->settings;
    Neuron* neuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;

    for (uint32_t blockId = brick->neuronBlockPos;
         blockId < brick->numberOfNeuronBlocks + brick->neuronBlockPos;
         ++blockId)
    {
        targetNeuronBlock = &neuronBlocks[blockId];
        for (uint32_t neuronId = 0; neuronId < NEURONS_PER_NEURONSECTION; ++neuronId) {
            neuron = &targetNeuronBlock->neurons[neuronId];
            neuron->potential /= clusterSettings->neuronCooldown;
            neuron->refractionTime = neuron->refractionTime >> 1;

            if (neuron->refractionTime == 0) {
                neuron->potential = clusterSettings->potentialOverflow * neuron->input;
                neuron->refractionTime = clusterSettings->refractionTime;
            }

            neuron->potential -= neuron->border;
            neuron->active = neuron->potential > 0.0f;
            neuron->input = 0.0f;
            neuron->potential = log2(neuron->potential + 1.0f);

            if constexpr (doTrain) {
                if (neuron->active != 0 && neuron->inUse == 0) {
                    SourceLocationPtr originLocation;
                    originLocation.blockId = blockId;
                    originLocation.neuronId = neuronId;
                    createNewSection(cluster, originLocation, 0.0f, HanamiRoot::cpuSynapseBlocks);
                }
            }
        }
    }
}

/**
 * @brief process all bricks and their content of a specific cluster
 *
 * @param cluster cluster to process
 * @param doTrain true to run a taining-process
 */
inline void
processCluster(Cluster& cluster, const bool doTrain)
{
    Brick* brick = nullptr;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;

    // process input-bricks
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isInputBrick) {
            if (doTrain) {
                processNeuronsOfInputBrick<true>(cluster, brick);
            }
            else {
                processNeuronsOfInputBrick<false>(cluster, brick);
            }
        }
    }

    // process normal- and output-bricks
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isInputBrick) {
            continue;
        }

        if (doTrain) {
            processSynapses<true>(cluster, brick);
            if (brick->isOutputBrick == false) {
                processNeurons<true>(cluster, brick);
            }
        }
        else {
            processSynapses<false>(cluster, brick);
            if (brick->isOutputBrick == false) {
                processNeurons<false>(cluster, brick);
            }
        }

        if (brick->isOutputBrick) {
            processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks);
        }
    }
}

#endif  // HANAMI_CORE_PROCESSING_H
