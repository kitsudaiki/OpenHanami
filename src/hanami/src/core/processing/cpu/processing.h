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
#include <core/cluster/cluster.h>
#include <core/cluster/objects.h>
#include <core/processing/cluster_resize.h>
#include <hanami_crypto/hashes.h>
#include <hanami_root.h>
#include <math.h>

#include <cmath>

/**
 * @brief initialize a new synpase
 *
 * @param synapse pointer to the synapse, which should be (re-) initialized
 * @param clusterSettings pointer to the cluster-settings
 * @param remainingW new weight for the synapse
 * @param randomSeed reference to the current seed of the randomizer
 */
inline void
createNewSynapse(Synapse* synapse,
                 const ClusterSettings* clusterSettings,
                 const float remainingW,
                 uint32_t& randomSeed)
{
    constexpr float randMax = static_cast<float>(RAND_MAX);
    constexpr float sigNeg = 0.5f;
    uint32_t signRand = 0;

    // set activation-border
    synapse->border = remainingW;

    // set initial active-counter for reduction-process
    synapse->activeCounter = 5;

    // set target neuron
    randomSeed = Hanami::pcg_hash(randomSeed);
    synapse->targetNeuronId = static_cast<uint16_t>(randomSeed % NEURONS_PER_NEURONBLOCK);

    randomSeed = Hanami::pcg_hash(randomSeed);
    synapse->weight = (static_cast<float>(randomSeed) / randMax) / 10.0f;

    // update weight with sign
    randomSeed = Hanami::pcg_hash(randomSeed);
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
 * @param clusterSettings pointer to cluster-settings
 * @param randomSeed reference to the current seed of the randomizer
 */
template <bool doTrain>
inline void
synapseProcessingBackward(Cluster& cluster,
                          SynapseSection* synapseSection,
                          Connection* connection,
                          NeuronBlock* targetNeuronBlock,
                          ClusterSettings* clusterSettings,
                          uint32_t& randomSeed)
{
    float potential = connection->potential - connection->lowerBound;
    float val = 0.0f;
    uint8_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    float halfPotential = 0.0f;
    const bool isAbleToCreate = connection->origin.isInput || cluster.enableCreation;

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
                    createNewSynapse(synapse, clusterSettings, potential, randomSeed);
                    cluster.enableCreation = true;
                }
            }
            if (isAbleToCreate && potential < (0.5f + connection->tollerance) * synapse->border
                && potential > (0.5f - connection->tollerance) * synapse->border)
            {
                synapse->border /= 1.5f;
                synapse->weight /= 1.5f;
                connection->tollerance /= 1.2f;
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
        if (potential > 0.00001f && isAbleToCreate) {
            connection->splitValue = halfPotential;
        }
    }
}

/**
 * @brief process all synapes of a hexagon
 *
 * @param cluster cluster, where the hexagon belongs to
 * @param hexagon pointer to current hexagon
 * @param blockId id of the current block within the hexagon
 */
template <bool doTrain>
inline void
processSynapses(Cluster& cluster, Hexagon* hexagon, const uint32_t blockId)
{
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(hexagon->attachedHost->synapseBlocks);
    SynapseBlock* synapseBlock = nullptr;
    SynapseSection* section = nullptr;
    NeuronBlock* neuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    Connection* connection = nullptr;
    ClusterSettings* clusterSettings = &cluster.clusterHeader.settings;
    uint32_t randomeSeed = rand();

    if (blockId >= hexagon->header.numberOfBlocks) {
        return;
    }

    connectionBlock = &hexagon->connectionBlocks[blockId];
    neuronBlock = &hexagon->neuronBlocks[blockId];
    synapseBlock = &synapseBlocks[hexagon->synapseBlockLinks[blockId]];

    for (uint32_t i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
        connection = &connectionBlock->connections[i];
        if (connection->origin.blockId != UNINIT_STATE_16 && connection->potential > 0.0f) {
            randomeSeed += (blockId * NUMBER_OF_SYNAPSESECTION) + i;
            section = &synapseBlock->sections[i];

            synapseProcessingBackward<doTrain>(
                cluster, section, connection, neuronBlock, clusterSettings, randomeSeed);
        }
    }
}

/**
 * @brief process all neurons of a hexagon
 *
 * @param cluster cluster, where the hexagon belongs to
 * @param hexagon pointer to current hexagon
 * @param blockId id of the current block within the hexagon
 */
template <bool doTrain>
inline void
processNeurons(Cluster& cluster, Hexagon* hexagon, const uint32_t blockId)
{
    ClusterSettings* clusterSettings = &cluster.clusterHeader.settings;
    NeuronBlock* neuronBlock = nullptr;
    Neuron* neuron = nullptr;

    neuronBlock = &hexagon->neuronBlocks[blockId];
    for (uint8_t neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
        neuron = &neuronBlock->neurons[neuronId];
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
            neuron->delta = 0.0f;
            if (neuron->active != 0 && neuron->inUse == 0) {
                createNewSection(cluster, hexagon, neuron, blockId, neuronId);
            }
        }
    }
}

/**
 * @brief process input-neurons
 *
 * @param cluster reference to current cluster
 * @param inputInterface pointer to connected input-interface
 * @param hexagon pointer to current hexagon
 */
template <bool doTrain>
inline void
processNeuronsOfInputHexagon(Cluster& cluster, InputInterface* inputInterface, Hexagon* hexagon)
{
    Neuron* neuron = nullptr;
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
                    createNewSection(cluster, hexagon, neuron, blockId, neuronId);
                }
            }
            counter++;
        }
        blockId++;
    }
}

/**
 * @brief process output-nodes
 *
 * @param hexagon current hexagon
 * @param randomSeed current seed for random-generation
 */
template <bool doTrain>
inline void
processNeuronsOfOutputHexagon(Hexagon* hexagon, uint32_t randomSeed)
{
    Neuron* neuron = nullptr;
    OutputTargetLocationPtr* target = nullptr;
    float weightSum = 0.0f;
    bool found = false;

    for (NeuronBlock& neuronBlock : hexagon->neuronBlocks) {
        for (uint64_t j = 0; j < NEURONS_PER_NEURONBLOCK; ++j) {
            neuron = &neuronBlock.neurons[j];
            neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->input));
            neuron->input = 0.0f;
        }
    }

    for (OutputNeuron& out : hexagon->outputInterface->outputNeurons) {
        weightSum = 0.0f;

        for (uint8_t j = 0; j < NUMBER_OF_OUTPUT_CONNECTIONS; ++j) {
            target = &out.targets[j];

            if constexpr (doTrain) {
                found = false;
                randomSeed = Hanami::pcg_hash(randomSeed);
                if (found == false && target->blockId == UNINIT_STATE_16 && out.exprectedVal > 0.0
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

        out.outputVal = 0.0f;
        if (weightSum != 0.0f) {
            out.outputVal = 1.0f / (1.0f + exp(-1.0f * weightSum));
        }
        // std::cout << out->outputVal << " : " << out->exprectedVal << std::endl;
    }
    // std::cout << "-------------------------------------" << std::endl;
}

/**
 * @brief process cluster and train it be creating new synapses
 *
 * @param cluster pointer to cluster to process
 * @param hexagonId id of the hexagon to process
 * @param blockId id of the block within the hexagon
 * @param doTrain true to run trainging-process
 */
inline void
processClusterForward(Cluster& cluster,
                      const uint32_t hexagonId,
                      const uint32_t blockId,
                      const bool doTrain)
{
    Hexagon* hexagon = &cluster.hexagons[hexagonId];

    if (hexagon->header.isInputHexagon) {
        return;
    }

    if (doTrain) {
        processSynapses<true>(cluster, hexagon, blockId);
        if (hexagon->header.isOutputHexagon == false) {
            processNeurons<true>(cluster, hexagon, blockId);
        }
    }
    else {
        processSynapses<false>(cluster, hexagon, blockId);
        if (hexagon->header.isOutputHexagon == false) {
            processNeurons<false>(cluster, hexagon, blockId);
        }
    }
}

/**
 * @brief handle input-hexagons by applying input-values to the input-neurons
 *
 * @param cluster pointer to cluster to process
 * @param doTrain true to run trainging-process
 */
inline void
processInput(Cluster& cluster, Hexagon* hexagon, const bool doTrain)
{
    assert(hexagon->inputInterface != nullptr);

    if (doTrain) {
        processNeuronsOfInputHexagon<true>(cluster, hexagon->inputInterface, hexagon);
    }
    else {
        processNeuronsOfInputHexagon<false>(cluster, hexagon->inputInterface, hexagon);
    }
}

/**
 * @brief handleConnectionBlocksForward
 * @param cluster
 * @param hexagon
 */
inline void
processConnectionBlocksForward(Cluster& cluster, Hexagon* hexagon)
{
    Connection* connection = nullptr;
    SourceLocation sourceLoc;

    for (ConnectionBlock& connectionBlock : hexagon->connectionBlocks) {
        for (uint32_t i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
            connection = &connectionBlock.connections[i];
            if (connection->origin.blockId == UNINIT_STATE_16) {
                continue;
            }

            // inputConnected = scon->origin.isInput;
            sourceLoc = getSourceNeuron(connection->origin, &cluster.hexagons[0]);
            connection->potential
                = static_cast<float>(sourceLoc.neuron->active) * sourceLoc.neuron->potential;
        }
    }
}

#endif  // HANAMI_CORE_PROCESSING_H
