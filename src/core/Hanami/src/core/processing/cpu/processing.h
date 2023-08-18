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

#include <common.h>
#include <math.h>
#include <cmath>

#include <hanami_root.h>
#include <core/cluster/cluster.h>
#include <core/processing/objects.h>
#include <core/processing/section_update.h>
#include <api/websocket/cluster_io.h>

/**
 * @brief get position of the highest output-position
 *
 * @param segment output-segment to check
 *
 * @return position of the highest output.
 */
inline uint32_t
getHighestOutput(const Cluster &cluster)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for(uint32_t outputNeuronId = 0;
        outputNeuronId < cluster.clusterHeader->outputValues.count;
        outputNeuronId++)
    {
        value = cluster.outputValues[outputNeuronId];
        if(value > hightest)
        {
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
                 const SegmentSettings* segmentSettings,
                 const float remainingW)
{
    const uint32_t* randomValues = HanamiRoot::m_randomValues;
    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = segmentSettings->signNeg;

    // set activation-border
    synapse->border = remainingW;

    // set target neuron
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = static_cast<uint16_t>(randomValues[block->randomPos]
                              % block->numberOfNeurons);


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
inline void
synapseProcessing(Cluster &cluster,
                  Synapse* section,
                  SynapseConnection* connection,
                  LocationPtr* currentLocation,
                  const float outH,
                  NeuronBlock* neuronBlocks,
                  SynapseBlock* synapseBlocks,
                  SynapseConnection* synapseConnections,
                  SegmentSettings* segmentSettings)
{
    uint32_t pos = 0;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    uint8_t active = 0;
    float counter = outH - connection->offset[currentLocation->sectionId];
    NeuronBlock* neuronBlock = &neuronBlocks[connection->targetNeuronBlockId];

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && counter > 0.01f)
    {
        synapse = &section[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16) {
            createNewSynapse(neuronBlock, synapse, segmentSettings, counter);
        }

        if(synapse->border > 2.0f * counter
                && pos < SYNAPSES_PER_SYNAPSESECTION-2)
        {
            const float val = synapse->border / 2.0f;
            section[pos + 1].border += val;
            synapse->border -= val;
        }

        // update target-neuron
        targetNeuron = &neuronBlock->neurons[synapse->targetNeuronId];
        targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 126);

        // update loop-counter
        counter -= synapse->border;
        pos++;
    }

    LocationPtr* targetLocation = &connection->next[currentLocation->sectionId];
    if(counter > 0.01f
            && targetLocation->sectionId == UNINIT_STATE_16)
    {
        const float newOffset = (outH - counter) + connection->offset[currentLocation->sectionId];
        createNewSection(cluster, connection->origin[currentLocation->sectionId], newOffset);
        targetLocation = &connection->next[currentLocation->sectionId];
    }

    if(targetLocation->sectionId != UNINIT_STATE_16)
    {
        Synapse* nextSection = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
        SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
        synapseProcessing(cluster,
                          nextSection,
                          nextConnection,
                          targetLocation,
                          outH,
                          neuronBlocks,
                          synapseBlocks,
                          synapseConnections,
                          segmentSettings);
    }
}

/**
 * @brief process only a single neuron
 */
inline void
processSingleNeuron(Cluster &cluster,
                    Neuron* neuron,
                    const uint32_t blockId,
                    const uint32_t neuronIdInBlock,
                    NeuronBlock* neuronBlocks,
                    SynapseBlock* synapseBlocks,
                    SynapseConnection* synapseConnections,
                    SegmentSettings* segmentSettings)
{
    // handle active-state
    if(neuron->active == 0) {
        return;
    }

    LocationPtr* targetLocation = &neuron->target;
    if(targetLocation->blockId == UNINIT_STATE_32)
    {
        LocationPtr sourceLocation;
        sourceLocation.blockId = blockId;
        sourceLocation.sectionId = neuronIdInBlock;
        if(createNewSection(cluster, sourceLocation, 0.0f) == false) {
            return;
        }
        targetLocation = &neuron->target;
    }

    Synapse* nextSection = synapseBlocks[targetLocation->blockId].synapses[targetLocation->sectionId];
    SynapseConnection* nextConnection = &synapseConnections[targetLocation->blockId];
    synapseProcessing(cluster,
                      nextSection,
                      nextConnection,
                      targetLocation,
                      neuron->potential,
                      neuronBlocks,
                      synapseBlocks,
                      synapseConnections,
                      segmentSettings);
}

/**
 * @brief process output brick
 */
inline void
processNeuronsOfOutputBrick(Cluster &cluster,
                            const Brick* brick,
                            float* outputValues,
                            NeuronBlock* neuronBlocks,
                            SegmentSettings* segmentSettings)
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
            neuron->potential = segmentSettings->potentialOverflow * neuron->input;
            if(neuron->potential != 0.0f) {
                neuron->potential = 1.0f / (1.0f + exp(-1.0f * neuron->potential));
            }
            //std::cout<<"neuron->potential: "<<neuron->potential<<std::endl;
            outputValues[counter] = neuron->potential;
            neuron->input = 0.0f;
            counter++;
        }

        // send output back if a client-connection is set
        if(cluster.msgClient != nullptr
                && cluster.mode == ClusterProcessingMode::NORMAL_MODE)
        {
             sendClusterOutputMessage(&cluster);
        }
    }
}

/**
 * @brief process input brick
 */
inline void
processNeuronsOfInputBrick(Cluster &cluster,
                           const Brick* brick,
                           float* inputValues,
                           NeuronBlock* neuronBlocks,
                           SynapseBlock* synapseBlocks,
                           SynapseConnection* synapseConnections,
                           SegmentSettings* segmentSettings)
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

            processSingleNeuron(cluster,
                                neuron,
                                blockId,
                                neuronIdInBlock,
                                neuronBlocks,
                                synapseBlocks,
                                synapseConnections,
                                segmentSettings);

            counter++;
        }
    }
}

/**
 * @brief process normal internal brick
 */
inline void
processNeuronsOfNormalBrick(Cluster &cluster,
                            const Brick* brick,
                            NeuronBlock* neuronBlocks,
                            SynapseBlock* synapseBlocks,
                            SynapseConnection* synapseConnections,
                            SegmentSettings* segmentSettings)
{
    Neuron* neuron = nullptr;
    NeuronBlock* blocks = nullptr;

    // iterate over all neurons within the brick
    for(uint32_t blockId = brick->brickBlockPos;
        blockId < brick->numberOfNeuronBlocks + brick->brickBlockPos;
        blockId++)
    {
        blocks = &neuronBlocks[blockId];
        for(uint32_t neuronIdInBlock = 0;
            neuronIdInBlock < blocks->numberOfNeurons;
            neuronIdInBlock++)
        {
            neuron = &blocks->neurons[neuronIdInBlock];

            neuron->potential /= segmentSettings->neuronCooldown;
            neuron->refractionTime = neuron->refractionTime >> 1;

            if(neuron->refractionTime == 0)
            {
                neuron->potential = segmentSettings->potentialOverflow * neuron->input;
                neuron->refractionTime = segmentSettings->refractionTime;
            }

            // update neuron
            neuron->potential -= neuron->border;
            neuron->active = neuron->potential > 0.0f;
            neuron->input = 0.0f;
            neuron->potential = log2(neuron->potential + 1.0f);

            processSingleNeuron(cluster,
                                neuron,
                                blockId,
                                neuronIdInBlock,
                                neuronBlocks,
                                synapseBlocks,
                                synapseConnections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process all neurons within a segment
 */
inline void
prcessCoreSegment(Cluster &cluster)
{
    float* inputValues = cluster.inputValues;
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
    SynapseConnection* synapseConnections = cluster.synapseConnections;
    SynapseBlock* synapseBlocks = cluster.synapseBlocks;
    SegmentSettings* segmentSettings = cluster.clusterSettings;

    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        const uint32_t brickId = cluster.brickOrder[pos];
        Brick* brick = &cluster.bricks[brickId];
        if(brick->isInputBrick)
        {
            processNeuronsOfInputBrick(cluster,
                                       brick,
                                       inputValues,
                                       neuronBlocks,
                                       synapseBlocks,
                                       synapseConnections,
                                       segmentSettings);
        }
        else if(brick->isOutputBrick)
        {
            processNeuronsOfOutputBrick(cluster,
                                        brick,
                                        outputValues,
                                        neuronBlocks,
                                        segmentSettings);
        }
        else
        {
            processNeuronsOfNormalBrick(cluster,
                                        brick,
                                        neuronBlocks,
                                        synapseBlocks,
                                        synapseConnections,
                                        segmentSettings);
        }
    }
}

#endif // HANAMI_CORE_PROCESSING_H
