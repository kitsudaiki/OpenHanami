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

template <bool doTrain>
__device__ __forceinline__ void
synapseProcessingBackward(Synapse* section,
                          SynapseConnection* connection,
                          NeuronBlock* currentNeuronBlock,
                          NeuronBlock* neuronBlocks,
                          ClusterSettings* clusterSettings,
                          const uint* randomValues,
                          float* localMem)
{
    const uint tid = threadIdx.x;

    NeuronBlock* sourceNeuronBlock = &neuronBlocks[connection->origin[tid].blockId];
    Neuron* sourceNeuron = &sourceNeuronBlock->neurons[connection->origin[tid].sectionId];
    float counter = sourceNeuron->potential - connection->offset[tid];
    uint pos = 0;
    Synapse* synapse;
    const uint32_t randomPos = (currentNeuronBlock->randomPos + (threadIdx.x * blockIdx.x) + 1)
                               % (NUMBER_OF_RAND_VALUES - 3);

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && counter > 0.01f)
    {
        synapse = &section[pos];

        if constexpr (doTrain)
        {
            // create new synapse if necesarry and training is active
            if(synapse->targetNeuronId == UNINIT_STATE_16)
            {
                // set activation-border
                synapse->border = counter;
                synapse->targetNeuronId = (ushort)(randomValues[randomPos]
                                          % currentNeuronBlock->numberOfNeurons);
                synapse->weight = ((float)(randomValues[randomPos + 1]) / (float)(RAND_MAX)) / 10.0f;
                const uint signRand = randomValues[randomPos + 2] % 1000;
                synapse->weight *= (float)(1.0f - (1000.0f * clusterSettings->signNeg > signRand) * 2);
            }

            if(synapse->border > 2.0f * counter
                    && pos < SYNAPSES_PER_SYNAPSESECTION-2)
            {
                const float val = synapse->border / 2.0f;
                section[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        // update target-neuron
        if(synapse->targetNeuronId != UNINIT_STATE_16)
        {
            // update target-neuron
            if(counter >= synapse->border) {
                localMem[synapse->targetNeuronId] += synapse->weight;
            } else {
                localMem[synapse->targetNeuronId] += synapse->weight * ((1.0f / synapse->border) * counter);
            }
        }

        // update loop-counter
        counter -= synapse->border;
        ++pos;
    }

    if constexpr (doTrain)
    {
        sourceNeuron->isNew = counter > 0.01f
                              && connection->next[tid].blockId == UNINIT_STATE_32;
        sourceNeuron->newOffset = (sourceNeuron->potential - counter)
                                  + connection->offset[tid];
    }
}

//==================================================================================================

template <bool doTrain>
__global__ void
processNeuronsOfNormalBrick(NeuronBlock* neuronBlocks,
                            SynapseBlock* synapseBlocks,
                            SynapseConnection* synapseConnections,
                            ClusterSettings* clusterSettings,
                            const uint32_t* randomValues,
                            const uint32_t neuronBlockPos,
                            const bool isOutputBrick)
{
    const uint tid = threadIdx.x;
    __shared__ float tempVal[64][64];
    for(uint i = 0; i < 64; ++i){
        tempVal[tid][i] = 0.0f;
    }

    NeuronBlock* currentNeuronBlock = &neuronBlocks[neuronBlockPos + blockIdx.x];
    currentNeuronBlock->randomPos = (currentNeuronBlock->randomPos + blockIdx.x + 1) % NUMBER_OF_RAND_VALUES;

    uint32_t synapseBlockId = currentNeuronBlock->backwardNextId;
    while(synapseBlockId != UNINIT_STATE_32)
    {
        // process synapse-sections
        Synapse* section = synapseBlocks[synapseBlockId].synapses[tid];
        SynapseConnection* connection = &synapseConnections[synapseBlockId];

        if(connection->origin[tid].blockId != UNINIT_STATE_32)
        {
            synapseProcessingBackward<doTrain>(section,
                                               connection,
                                               currentNeuronBlock,
                                               neuronBlocks,
                                               clusterSettings,
                                               randomValues,
                                               tempVal[tid]);
        }

        __syncthreads();

        synapseBlockId = connection->backwardNextId;
    }

    if(tid < currentNeuronBlock->numberOfNeurons)
    {
        Neuron* neuron = &currentNeuronBlock->neurons[tid];
        for(uint i = 0; i < 64; ++i) {
            neuron->input += tempVal[i][tid];
        }
    }

    if(isOutputBrick == false
            && tid < currentNeuronBlock->numberOfNeurons)
    {
        Neuron* neuron = &currentNeuronBlock->neurons[tid];

        neuron->potential /= clusterSettings->neuronCooldown;
        neuron->refractionTime = neuron->refractionTime >> 1;

        if(neuron->refractionTime == 0)
        {
            neuron->potential = clusterSettings->potentialOverflow * neuron->input;
            neuron->refractionTime = clusterSettings->refractionTime;
        }

        // update neuron
        neuron->potential -= neuron->border;
        neuron->active = neuron->potential > 0.0f;
        neuron->input = 0.0f;
        neuron->potential = log2(neuron->potential + 1.0f);

        // handle active-state
        neuron->isNew = neuron->active != 0
                && neuron->target.blockId == UNINIT_STATE_32;
        neuron->newOffset = 0.0f;
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief correct weight of synapses within a cluster
 */
__global__ void
reweightCoreSegmentKernel(NeuronBlock* neuronBlocks,
                          SynapseBlock* synapseBlocks,
                          SynapseConnection* synapseConnections,
                          ClusterSettings* clusterSettings,
                          const uint32_t neuronSectionPos)
{
    __shared__ float tempVal[64];
    __shared__ LocationPtr loc[64];

    const uint tid = threadIdx.x;
    const float trainValue = 0.05f;
    float counter;

    NeuronBlock* currentNeuronBlock = &neuronBlocks[neuronSectionPos + blockIdx.x];
    if(tid < currentNeuronBlock->numberOfNeurons)
    {
        Neuron* sourceNeuron = &currentNeuronBlock->neurons[tid];
        loc[tid] = sourceNeuron->target;
        sourceNeuron->delta = 0.0f;

        if(loc[tid].blockId != UNINIT_STATE_32
                && sourceNeuron->active)
        {
            tempVal[tid] = 0.0f;

            while(loc[tid].blockId != UNINIT_STATE_32)
            {
                Synapse* section = synapseBlocks[loc[tid].blockId].synapses[loc[tid].sectionId];
                SynapseConnection* connection = &synapseConnections[loc[tid].blockId];
                NeuronBlock* targetNeuronSection = &neuronBlocks[connection->targetNeuronBlockId];

                tempVal[tid] = sourceNeuron->delta;
                counter = sourceNeuron->potential - connection->offset[tid];

                // iterate over all synapses in the section
                for(uint32_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; ++pos)
                {
                    Synapse* synapse = &section[pos];

                    const float update = static_cast<float>(counter > 0.01f && synapse->targetNeuronId != UNINIT_STATE_16);
                    Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId % UNINIT_STATE_16];
                    tempVal[tid] += targetNeuron->delta * synapse->weight * update;

                    synapse->weight -= trainValue * targetNeuron->delta * update;
                    counter -= synapse->border;
                }

                loc[tid] = connection->next[loc[tid].sectionId];
            }

            tempVal[tid] *= 1.4427f * pow(0.5f, sourceNeuron->potential);
            sourceNeuron->delta = tempVal[tid];
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
                  SynapseConnection* synapseConnections,
                  const uint32_t numberOfSynapseConnections,
                  uint32_t* randomValues)
{
    cudaMalloc(&gpuPointer->clusterSettings,    1                          * sizeof(ClusterSettings));
    cudaMalloc(&gpuPointer->neuronBlocks,       numberOfNeuronBlocks       * sizeof(NeuronBlock));
    cudaMalloc(&gpuPointer->synapseBlocks,      numberOfSynapseBlocks      * sizeof(SynapseBlock));
    cudaMalloc(&gpuPointer->synapseConnections, numberOfSynapseConnections * sizeof(SynapseConnection));
    cudaMalloc(&gpuPointer->randomValues,       NUMBER_OF_RAND_VALUES      * sizeof(uint32_t));

    cudaMemcpy(gpuPointer->clusterSettings,    clusterSettings,    1                          * sizeof(ClusterSettings),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronBlocks,       neuronBlocks,       numberOfNeuronBlocks       * sizeof(NeuronBlock),       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseBlocks,      synapseBlocks,      numberOfSynapseBlocks      * sizeof(SynapseBlock),      cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseConnections, synapseConnections, numberOfSynapseConnections * sizeof(SynapseConnection), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->randomValues,       randomValues,       NUMBER_OF_RAND_VALUES      * sizeof(uint32_t),          cudaMemcpyHostToDevice);
}



extern "C"
void
processing_CUDA(PointerHandler* gpuPointer,
                uint32_t* brickOrder,
                Brick* bricks,
                float* inputValues,
                float* outputValues,
                const uint32_t numberOfBricks,
                NeuronBlock* neuronBlocks,
                const uint32_t numberOfNeuronBlocks,
                SynapseBlock* synapseBlocks,
                const uint32_t numberOfSynapseBlocks,
                const bool doTrain)
{
    for(uint32_t pos = 0; pos < numberOfBricks; ++pos)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick)
        {
            if(doTrain) {
                processNeuronsOfInputBrickBackward<true>(brick, inputValues, neuronBlocks);
            } else {
                processNeuronsOfInputBrickBackward<false>(brick, inputValues, neuronBlocks);
            }
        }
    }

    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    for(uint32_t pos = 0; pos < numberOfBricks; ++pos)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick == false)
        {
            if(doTrain)
            {
                processNeuronsOfNormalBrick<true><<<brick->numberOfNeuronBlocks, 64>>>(
                    gpuPointer->neuronBlocks,
                    gpuPointer->synapseBlocks,
                    gpuPointer->synapseConnections,
                    gpuPointer->clusterSettings,
                    gpuPointer->randomValues,
                    brick->brickBlockPos,
                    brick->isOutputBrick);
            }
            else
            {
                processNeuronsOfNormalBrick<false><<<brick->numberOfNeuronBlocks, 64>>>(
                    gpuPointer->neuronBlocks,
                    gpuPointer->synapseBlocks,
                    gpuPointer->synapseConnections,
                    gpuPointer->clusterSettings,
                    gpuPointer->randomValues,
                    brick->brickBlockPos,
                    brick->isOutputBrick);
            }
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);

    for(uint32_t pos = 0; pos < numberOfBricks; ++pos)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isOutputBrick) {
            processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks);
        }
    }
}

extern "C"
void
backpropagation_CUDA(PointerHandler* gpuPointer,
                     uint32_t* brickOrder,
                     Brick* bricks,
                     float* outputValues,
                     float* expectedValues,
                     const uint32_t numberOfBricks,
                     NeuronBlock* neuronBlocks,
                     const uint32_t numberOfNeuronBlocks,
                     ClusterSettings* settings)
{
    for(uint32_t pos = 0; pos < numberOfBricks; ++pos)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isOutputBrick)
        {
            if(backpropagateOutput(brick,
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

    for(int32_t pos = numberOfBricks - 1; pos >= 0; --pos)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isOutputBrick == false)
        {
            reweightCoreSegmentKernel<<<brick->numberOfNeuronBlocks, 64>>>(
                    gpuPointer->neuronBlocks,
                    gpuPointer->synapseBlocks,
                    gpuPointer->synapseConnections,
                    gpuPointer->clusterSettings,
                    brick->brickBlockPos);
        }
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
            SynapseConnection* synapseConnections,
            const uint32_t numberOfSynapseConnections)
{
    cudaMemcpy(gpuPointer->synapseConnections,
               synapseConnections,
               numberOfSynapseConnections * sizeof(SynapseConnection),
               cudaMemcpyHostToDevice);

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

