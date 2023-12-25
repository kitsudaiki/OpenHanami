/**
 * @file        gpu_kernel.cu
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2019 Tobias Anker
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

#include <iostream>
#include <chrono>
#include <math.h>

#include "../objects.h"
#include "../cluster_io_functions.h"


//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief createNewSynapse
 * @param block
 * @param synapse
 * @param clusterSettings
 * @param remainingW
 */
__device__ __forceinline__ void
createNewSynapse(NeuronBlock* block,
                 Synapse* synapse,
                 const ClusterSettings* clusterSettings,
                 const float remainingW,
                 const uint32_t* randomValues)
{
    uint32_t randomPos = (block->randomPos + (threadIdx.x * blockIdx.x) + 1)
                               % (NUMBER_OF_RAND_VALUES - 5);
    block->randomPos = randomPos;

    const float randMax = static_cast<float>(RAND_MAX);
    uint32_t signRand = 0;
    const float sigNeg = clusterSettings->signNeg;

    // set activation-border
    synapse->border = remainingW;

    // set target neuron
    randomPos = (randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId
        = static_cast<uint16_t>(randomValues[randomPos] % block->numberOfNeurons);

    randomPos = (randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = (static_cast<float>(randomValues[randomPos]) / randMax) / 10.0f;

    // update weight with sign
    randomPos = (randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[randomPos] % 1000;
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
__device__ __forceinline__ void
synapseProcessingBackward(SynapseSection* section,
                          SynapseConnection* connection,
                          NeuronBlock* targetNeuronBlock,
                          Neuron* sourceNeuron,
                          const SourceLocationPtr originLocation,
                          ClusterSettings* clusterSettings,
                          const uint* randomValues,
                          float* localMem)
{
    float counter = sourceNeuron->potential - connection->offset;
    uint pos = 0;
    Synapse* synapse = nullptr;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        synapse = &section->synapses[pos];

        if constexpr (doTrain) {
            // create new synapse if necesarry and training is active
            if (synapse->targetNeuronId == UNINIT_STATE_16) {
                createNewSynapse(targetNeuronBlock, synapse, clusterSettings, counter, randomValues);
            }

            if (synapse->border > 2.0f * counter && pos < SYNAPSES_PER_SYNAPSESECTION - 2) {
                const float val = synapse->border / 2.0f;
                section->synapses[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            // update target-neuron
            if (counter >= synapse->border) {
                localMem[synapse->targetNeuronId] += synapse->weight;
            } else {
                localMem[synapse->targetNeuronId] += synapse->weight * ((1.0f / synapse->border) * counter);
            }
        }

        // update loop-counter
        counter -= synapse->border;
        ++pos;
    }

    if constexpr (doTrain) {
        sourceNeuron->isNew = (counter > 0.01f
                               && section->hasNext == false) * (connection->origin.posInNeuron + 1);
        sourceNeuron->newOffset = (sourceNeuron->potential - counter) + connection->offset;
        section->hasNext = true;
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
__global__ void
processBrick(NeuronBlock* neuronBlocks,
             SynapseBlock* synapseBlocks,
             ConnectionBlock* connectionBlocks,
             ClusterSettings* clusterSettings,
             const uint32_t* randomValues,
             const uint32_t neuronBlockPos,
             const uint32_t numberConnectionBlocks,
             const uint32_t dimY,
             const bool isOutputBrick)
{
    SynapseConnection* scon = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    NeuronBlock* targetNeuronBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseSection* section = nullptr;
    const uint64_t tid = threadIdx.x;
    const uint64_t neuronBlockId = blockIdx.x + neuronBlockPos;

    // init temp-values
    __shared__ float tempVal[64][64];
    for (uint i = 0; i < 64; ++i){
        tempVal[tid][i] = 0.0f;
    }

    // process synapses
    for (int c = blockIdx.x; c < blockIdx.x + dimY; ++c) {
        connectionBlock = &connectionBlocks[c];
        scon = &connectionBlock->connections[tid];

        if (scon->origin.blockId != UNINIT_STATE_32) {
            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.sectionId];
            if (sourceNeuron->active != 0) {
                section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[tid];
                targetNeuronBlock = &neuronBlocks[neuronBlockId];

                synapseProcessingBackward<doTrain>(section,
                                                   scon,
                                                   targetNeuronBlock,
                                                   sourceNeuron,
                                                   scon->origin,
                                                   clusterSettings,
                                                   randomValues,
                                                   tempVal[tid]);
            }
        }
    }

    __syncthreads();

    // aggregate temp-values
    NeuronBlock* currentNeuronBlock = &neuronBlocks[neuronBlockId];
    Neuron* neuron = &currentNeuronBlock->neurons[tid];
    for (uint i = 0; i < 64; ++i) {
        neuron->input += tempVal[i][tid];
    }

    // process neurons
    if (isOutputBrick == false) {

        targetNeuronBlock = &neuronBlocks[neuronBlockId];
        neuron = &targetNeuronBlock->neurons[tid];
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
            neuron->isNew = neuron->active != 0 && neuron->inUse == 0;
            neuron->newOffset = 0.0f;
        }
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief backpropagateSection
 * @param section
 * @param connection
 * @param targetNeuronBlock
 * @param sourceNeuron
 */
__device__ __forceinline__ void
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     NeuronBlock* targetNeuronBlock,
                     Neuron* sourceNeuron)
{
    float counter = sourceNeuron->potential - connection->offset;
    uint pos = 0;
    Synapse* synapse;
    Neuron* targetNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    float delta = 0.0f;
    sourceNeuron->delta[connection->origin.posInNeuron] = 0.0f;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && counter > 0.01f) {
        // break look, if no more synapses to process
        synapse = &section->synapses[pos];
        if (synapse->targetNeuronId != UNINIT_STATE_16) {
            // update weight
            targetNeuron = &targetNeuronBlock->neurons[synapse->targetNeuronId];
            delta = targetNeuron->delta[0] * synapse->weight;
            if (counter < synapse->border) {
                delta *= (1.0f / synapse->border) * counter;
            }
            synapse->weight -= trainValue * targetNeuron->delta[0];
            sourceNeuron->delta[connection->origin.posInNeuron] += delta;

            counter -= synapse->border;
        }
        ++pos;
    }
}

/**
 * @brief backpropagateNeurons
 * @param brick
 * @param neuronBlocks
 * @param synapseBlocks
 */
__global__ void
backpropagateNeurons(NeuronBlock* neuronBlocks,
                     SynapseBlock* synapseBlocks,
                     ConnectionBlock* connectionBlocks,
                     const uint32_t neuronBlockPos,
                     const uint32_t numberConnectionBlocks,
                     const uint32_t dimY)
{
    NeuronBlock* sourceNeuronBlock = nullptr;
    Neuron* sourceNeuron = nullptr;
    SynapseSection* section = nullptr;
    ConnectionBlock* connectionBlock = nullptr;
    SynapseConnection* scon = nullptr;
    float delta;
    const uint64_t tid = threadIdx.x;
    const uint64_t neuronBlockId = blockIdx.x + neuronBlockPos;

    NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];
    Neuron* neuron = &targetNeuronBlock->neurons[tid];

    neuron = &targetNeuronBlock->neurons[tid];
    if (neuron->active) {
        // aggregate different delta-values
        delta = 0.0f;
        delta += neuron->delta[0];
        delta += neuron->delta[1];
        delta += neuron->delta[2];
        delta += neuron->delta[3];
        delta += neuron->delta[4];
        delta += neuron->delta[5];
        delta += neuron->delta[6];
        delta += neuron->delta[7];

        // calculate new delta-value for next iteration
        delta *= 1.4427f * pow(0.5f, neuron->potential);
        neuron->delta[0] = delta;
        neuron->delta[1] = 0.0f;
        neuron->delta[2] = 0.0f;
        neuron->delta[3] = 0.0f;
        neuron->delta[4] = 0.0f;
        neuron->delta[5] = 0.0f;
        neuron->delta[6] = 0.0f;
        neuron->delta[7] = 0.0f;
    }

    __syncthreads();

    for (int c = blockIdx.x; c < blockIdx.x + dimY; ++c) {
        connectionBlock = &connectionBlocks[c];
        scon = &connectionBlock->connections[tid];

        if (scon->origin.blockId != UNINIT_STATE_32) {
            sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.sectionId];
            section = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[tid];
            targetNeuronBlock = &neuronBlocks[neuronBlockId];

            backpropagateSection(section, scon, targetNeuronBlock, sourceNeuron);
        }
    }
}

//==================================================================================================

extern "C"
void
copyToDevice_CUDA(PointerHandler* gpuPointer,
                  ClusterSettings* clusterSettings,
                  NeuronBlock* neuronBlocks,
                  const uint32_t numberOfNeuronBlocks,
                  SynapseBlock* synapseBlocks,
                  const uint32_t numberOfSynapseBlocks,
                  Brick* bricks,
                  const uint32_t numberOfBricks,
                  uint32_t* randomValues)
{
    cudaMalloc(&gpuPointer->clusterSettings,    1                          * sizeof(ClusterSettings));
    cudaMalloc(&gpuPointer->neuronBlocks,       numberOfNeuronBlocks       * sizeof(NeuronBlock));
    cudaMalloc(&gpuPointer->synapseBlocks,      numberOfSynapseBlocks      * sizeof(SynapseBlock));
    cudaMalloc(&gpuPointer->randomValues,       NUMBER_OF_RAND_VALUES      * sizeof(uint32_t));

    cudaMemcpy(gpuPointer->clusterSettings,    clusterSettings,    1                      * sizeof(ClusterSettings),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronBlocks,       neuronBlocks,       numberOfNeuronBlocks   * sizeof(NeuronBlock),       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseBlocks,      synapseBlocks,      numberOfSynapseBlocks  * sizeof(SynapseBlock),      cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->randomValues,       randomValues,       NUMBER_OF_RAND_VALUES  * sizeof(uint32_t),          cudaMemcpyHostToDevice);

    gpuPointer->connectionBlocks.resize(numberOfBricks);
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        gpuPointer->connectionBlocks[brickId] = nullptr;
    }
}



extern "C"
void
processing_CUDA(PointerHandler* gpuPointer,
                Brick* bricks,
                float* inputValues,
                float* outputValues,
                const uint32_t numberOfBricks,
                NeuronBlock* neuronBlocks,
                const uint32_t numberOfNeuronBlocks,
                const bool doTrain)
{
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick == false) {
            continue;
        }

        if (doTrain) {
            processNeuronsOfInputBrickBackward<true>(brick, inputValues, neuronBlocks);
        } else {
            processNeuronsOfInputBrickBackward<false>(brick, inputValues, neuronBlocks);
        }
    }

    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick) {
            continue;
        }

        if (doTrain)
        {
            processBrick<true><<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                gpuPointer->randomValues,
                brick->neuronBlockPos,
                brick->connectionBlocks.size(),
                brick->dimY,
                brick->isOutputBrick);
        }
        else
        {
            processBrick<false><<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                gpuPointer->randomValues,
                brick->neuronBlockPos,
                brick->connectionBlocks.size(),
                brick->dimY,
                brick->isOutputBrick);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);

    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isOutputBrick == false) {
            continue;
        }

        processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks);
    }
}

extern "C"
void
backpropagation_CUDA(PointerHandler* gpuPointer,
                     Brick* bricks,
                     float* outputValues,
                     float* expectedValues,
                     const uint32_t numberOfBricks,
                     NeuronBlock* neuronBlocks,
                     const uint32_t numberOfNeuronBlocks,
                     ClusterSettings* settings)
{
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isOutputBrick)
        {
            if (backpropagateOutput(brick,
                                   neuronBlocks,
                                   outputValues,
                                   expectedValues,
                                   settings) == false)
            {
                return;
            }
        }
    }

    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick) {
            continue;
        }

        backpropagateNeurons<<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                brick->neuronBlockPos,
                brick->connectionBlocks.size(),
                brick->dimY);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
}

extern "C"
void
update_CUDA(PointerHandler* gpuPointer,
            NeuronBlock* neuronBlocks,
            const uint32_t numberOfNeuronBlocks,
            Brick* bricks,
            const uint32_t numberOfBricks)
{
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (gpuPointer->connectionBlocks[brickId] != nullptr)
        {
            cudaFree(gpuPointer->connectionBlocks[brickId]);
            gpuPointer->connectionBlocks[brickId] = nullptr;
        }

        cudaMalloc(&gpuPointer->connectionBlocks[brickId],
                   brick->connectionBlocks.size() * sizeof(ConnectionBlock));
        cudaMemcpy(gpuPointer->connectionBlocks[brickId],
                   &brick->connectionBlocks[0],
                   brick->connectionBlocks.size() * sizeof(ConnectionBlock),
                   cudaMemcpyHostToDevice);
    }

    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);
}

extern "C"
void
copyFromGpu_CUDA(PointerHandler* gpuPointer,
                 NeuronBlock* neuronBlocks,
                 const uint32_t numberOfNeuronBlocks,
                 SynapseBlock* synapseBlocks,
                 const uint32_t numberOfSynapseBlocks)
{
    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(synapseBlocks,
               gpuPointer->synapseBlocks,
               numberOfSynapseBlocks * sizeof(SynapseBlock),
               cudaMemcpyDeviceToHost);
}

