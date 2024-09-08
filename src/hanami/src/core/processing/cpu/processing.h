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
    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = 0.5f;

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
 * @param sourceNeuron pointer to source-neuron, which had triggered the section
 * @param originLocation location of the source-neuron to mark updates
 * @param clusterSettings pointer to cluster-settings
 * @param inputConnected true, if source-neuron belongs to an input-hexagon
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
    NeuronBlock* neuronBlocks = &hexagon->neuronBlocks[0];
    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    ClusterSettings* clusterSettings = &cluster.clusterHeader.settings;
    SynapseConnection* scon = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    SynapseSection* section = nullptr;
    Hexagon* sourceHexagon = nullptr;
    uint32_t randomeSeed = rand();
    const uint32_t dimX = hexagon->header.dimX;
    SourceLocation sourceLoc;

    bool inputConnected = false;

    if (blockId >= dimX) {
        return;
    }

    assert(blockId < hexagon->connectionBlocks.size());
    connectionBlock = &hexagon->connectionBlocks[blockId];
    const uint64_t synapseBlockLink = hexagon->synapseBlockLinks[blockId];

    for (uint32_t i = 0; i < NUMBER_OF_SYNAPSESECTION; ++i) {
        scon = &connectionBlock->connections[i];
        if (scon->origin.blockId == UNINIT_STATE_16) {
            continue;
        }

        inputConnected = scon->origin.isInput;
        sourceLoc = getSourceNeuron(scon->origin, &cluster.hexagons[0]);

        if (sourceLoc.neuron->active == 0) {
            continue;
        }

        section = &synapseBlocks[synapseBlockLink].sections[i];
        targetNeuronBlock = &neuronBlocks[blockId];
        randomeSeed += (blockId * NUMBER_OF_SYNAPSESECTION) + i;

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
    Neuron* neuron = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    const uint32_t neuronBlockId = blockId;

    targetNeuronBlock = &hexagon->neuronBlocks[neuronBlockId];
    for (uint8_t neuronId = 0; neuronId < NEURONS_PER_NEURONBLOCK; ++neuronId) {
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
            neuron->delta = 0.0f;
            neuron->isNew = neuron->active != 0 && neuron->inUse == 0;
            neuron->newLowerBound = 0.0f;
            neuron->potentialRange = std::numeric_limits<float>::max();
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

/**
 * @brief process output-nodes
 *
 * @param hexagons list of all hexagons
 * @param outputInterface connected output-interface
 * @param hexagonId current hexagon-id
 * @param randomSeed current seed for random-generation
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

#endif  // HANAMI_CORE_PROCESSING_H
