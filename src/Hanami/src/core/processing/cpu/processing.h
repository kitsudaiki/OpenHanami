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
#include <core/processing/cluster_resize.h>
#include <core/processing/objects.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief process all neurons of an input-brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick input-brick to process
 */
template <bool doTrain>
inline void
processNeuronsOfInputBrick(Cluster& cluster, InputInterface* inputInterface, Brick* brick)
{
    Neuron* neuron = nullptr;
    NeuronBlock* block = nullptr;
    uint32_t counter = 0;
    uint32_t blockId = 0;
    uint32_t neuronId = 0;

    // iterate over all neurons within the brick
    for (NeuronBlock& neuronBlock : brick->neuronBlocks) {
        for (neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
            neuron = &neuronBlock.neurons[neuronId];
            neuron->potential = inputInterface->inputNeurons[counter].value;
            neuron->active = neuron->potential > 0.0f;

            if constexpr (doTrain) {
                if (neuron->active != 0 && neuron->inUse == 0) {
                    SourceLocationPtr originLocation;
                    originLocation.brickId = brick->brickId;
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

/**
 * @brief initialize a new synpase
 *
 * @param block source-neuron-block, which is only used to hold the randamo-value
 * @param synapse pointer to the synapse, which should be (re-) initialized
 * @param clusterSettings pointer to the cluster-settings
 * @param remainingW new weight for the synapse
 * @param randomSeed reference to the current seed of the randomizer
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
    const float sigNeg = 0.5f;

    // set activation-border
    synapse->border = remainingW;

    // set initial active-counter for reduction-process
    synapse->activeCounter = 5;

    // set target neuron
    randomSeed = pcg_hash(randomSeed);
    synapse->targetNeuronId = static_cast<uint16_t>(randomSeed % NEURONS_PER_NEURONBLOCK);

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
 * @param randomSeed reference to the current seed of the randomizer
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
                          const bool inputConnected,
                          uint32_t& randomSeed)
{
    float potential = sourceNeuron->potential - connection->lowerBound;
    float val = 0.0f;
    uint8_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    bool condition = false;
    float halfPotential = 0.0f;
    const bool isAbleToCreate = inputConnected || cluster.enableCreation;

    if (potential > connection->potentialRange) {
        potential = connection->potentialRange;
    }

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && potential > 0.00001f) {
        synapse = &synapseSection->synapses[pos];

        if constexpr (doTrain) {
            if (isAbleToCreate) {
                // create new synapse if necesarry and training is active
                if (synapse->targetNeuronId == UNINIT_STATE_8) {
                    createNewSynapse(
                        targetNeuronBlock, synapse, clusterSettings, potential, randomSeed);
                    cluster.enableCreation = true;
                }
            }
            if (isAbleToCreate && potential < (0.5f + synapseSection->tollerance) * synapse->border
                && potential > (0.5f - synapseSection->tollerance) * synapse->border)
            {
                synapse->border /= 1.5f;
                synapse->weight /= 1.5f;
                synapseSection->tollerance /= 1.2f;
                cluster.enableCreation = true;
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
        halfPotential
            += static_cast<float>(pos < SYNAPSES_PER_SYNAPSESECTION / 2) * synapse->border;
        potential -= synapse->border;
        ++pos;
    }

    if constexpr (doTrain) {
        sourceNeuron->isNew = false;
        if (potential > 0.00001f && isAbleToCreate) {
            sourceNeuron->isNew = true;
            sourceNeuron->newLowerBound = connection->lowerBound + halfPotential;
            sourceNeuron->potentialRange = connection->potentialRange - halfPotential;
            connection->potentialRange = halfPotential;
        }
    }
}

/**
 * @brief process all synapes of a brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick pointer to current brick
 * @param blockId id of the current block within the brick
 */
template <bool doTrain>
inline void
processSynapses(Cluster& cluster, Brick* brick, const uint32_t blockId)
{
    NeuronBlock* neuronBlocks = &brick->neuronBlocks[0];
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    ClusterSettings* clusterSettings = &cluster.clusterHeader.settings;
    SynapseConnection* scon = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    SynapseSection* section = nullptr;
    Brick* sourceBrick = nullptr;
    uint32_t randomeSeed = rand();
    const uint32_t dimY = brick->dimY;
    const uint32_t dimX = brick->dimX;
    SourceLocation sourceLoc;
    uint32_t c = 0;
    uint32_t i = 0;

    bool inputConnected = false;

    if (blockId >= dimX) {
        return;
    }

    for (c = blockId * dimY; c < (blockId * dimY) + dimY; ++c) {
        assert(c < brick->connectionBlocks.size());
        connectionBlock = &brick->connectionBlocks[c];

        for (i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
            scon = &connectionBlock->connections[i];
            if (scon->origin.blockId == UNINIT_STATE_16) {
                continue;
            }

            inputConnected = scon->origin.isInput;
            sourceLoc = getSourceNeuron(scon->origin, &cluster.bricks[0]);

            if (sourceLoc.neuron->active == 0) {
                continue;
            }

            section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[i];
            targetNeuronBlock = &neuronBlocks[(c / brick->dimY)];
            randomeSeed += (c * NUMBER_OF_SYNAPSESECTION) + i;

            synapseProcessingBackward<doTrain>(cluster,
                                               section,
                                               scon,
                                               targetNeuronBlock,
                                               sourceLoc.neuron,
                                               scon->origin,
                                               clusterSettings,
                                               inputConnected,
                                               randomeSeed);
        }
    }
}

/**
 * @brief process all neurons of a brick
 *
 * @param cluster cluster, where the brick belongs to
 * @param brick pointer to current brick
 * @param blockId id of the current block within the brick
 */
template <bool doTrain>
inline void
processNeurons(Cluster& cluster, Brick* brick, const uint32_t blockId)
{
    ClusterSettings* clusterSettings = &cluster.clusterHeader.settings;
    Neuron* neuron = nullptr;
    TempNeuron* tempNeuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    TempNeuronBlock* tempNeuronBlock = nullptr;
    const uint32_t neuronBlockId = blockId;

    targetNeuronBlock = &brick->neuronBlocks[neuronBlockId];
    if constexpr (doTrain) {
        tempNeuronBlock = &brick->tempNeuronBlocks[neuronBlockId];
    }
    for (uint32_t neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
        neuron = &targetNeuronBlock->neurons[neuronId];
        neuron->potential /= clusterSettings->neuronCooldown;
        neuron->refractoryTime = neuron->refractoryTime >> 1;

        if (neuron->refractoryTime == 0) {
            neuron->potential += clusterSettings->potentialOverflow * neuron->input;
            neuron->refractoryTime = clusterSettings->refractoryTime;
        }

        neuron->potential -= neuron->border;
        neuron->active = neuron->potential > 0.0f;
        neuron->potential = static_cast<float>(neuron->active) * neuron->potential;
        neuron->input = 0.0f;
        neuron->potential = log2(neuron->potential + 1.0f);

        if constexpr (doTrain) {
            tempNeuron = &tempNeuronBlock->neurons[neuronId];
            tempNeuron->delta[0] = 0.0f;

            neuron->isNew = neuron->active != 0 && neuron->inUse == 0;
            neuron->newLowerBound = 0.0f;
            neuron->potentialRange = std::numeric_limits<float>::max();
        }
    }
}

#endif  // HANAMI_CORE_PROCESSING_H
