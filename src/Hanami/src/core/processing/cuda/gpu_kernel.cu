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
#include <cuda_runtime.h>

#include "../objects.h"
#include "../cluster_io_functions.h"

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief initialize a new specific synapse
 */
__device__ __forceinline__ void
createNewSynapse(NeuronBlock* block,
                 Synapse* synapse,
                 const ClusterSettings* clusterSettings,
                 const float remainingW,
                 const uint* randomValues)
{
    // set activation-border
    synapse->border = remainingW;

    // set target neuron
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = (ushort)(randomValues[block->randomPos]
                              % block->numberOfNeurons);


    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = ((float)(randomValues[block->randomPos]) / (float)(RAND_MAX)) / 10.0f;

    // update weight with sign
    block->randomPos = (block->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    const uint signRand = randomValues[block->randomPos] % 1000;
    synapse->weight *= (float)(1.0f - (1000.0f * clusterSettings->signNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

//==================================================================================================

__device__ __forceinline__ void
synapseProcessingBackward(Synapse* section,
                          SynapseConnection* connection,
                          NeuronBlock* currentNeuronBlock,
                          NeuronBlock* neuronBlocks,
                          ClusterSettings* clusterSettings,
                          const uint* randomValues,
                          float* localMem)
{
    if(connection->origin[threadIdx.x].blockId != UNINIT_STATE_32)
    {
        NeuronBlock* sourceNeuronBlock = &neuronBlocks[connection->origin[threadIdx.x].blockId];
        Neuron* sourceNeuron = &sourceNeuronBlock->neurons[connection->origin[threadIdx.x].sectionId];
        float counter = sourceNeuron->potential - connection->offset[threadIdx.x];
        uint pos = 0;
        // iterate over all synapses in the section
        while(pos < SYNAPSES_PER_SYNAPSESECTION
              && counter > 0.01f)
        {
            Synapse* synapse = &section[pos];

            // create new synapse if necesarry and training is active
            if(synapse->targetNeuronId == UNINIT_STATE_16) {
                createNewSynapse(currentNeuronBlock, synapse, clusterSettings, counter, randomValues);
            }

            if(synapse->border > 2.0f * counter
                    && pos < SYNAPSES_PER_SYNAPSESECTION-2)
            {
                const float val = synapse->border / 2.0f;
                section[pos + 1].border += val;
                synapse->border -= val;
            }

            // update target-neuron
            Neuron* targetNeuron = &currentNeuronBlock->neurons[synapse->targetNeuronId];
            //targetNeuron->input += synapse->weight;
            localMem[synapse->targetNeuronId] += synapse->weight;

            // update active-counter
            const uint8_t active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
            synapse->activeCounter += active * (uint8_t)(synapse->activeCounter < 126);

            // update loop-counter
            counter -= synapse->border;
            pos++;
        }

        sourceNeuron->isNew = counter > 0.01f
                              && connection->next[threadIdx.x].blockId == UNINIT_STATE_32;
        sourceNeuron->newOffset = (sourceNeuron->potential - counter)
                                  + connection->offset[threadIdx.x];
    }
}

//==================================================================================================

__global__ void
prcessCoreSegmentKernel(NeuronBlock* neuronBlocks,
                        SynapseBlock* synapseBlocks,
                        SynapseConnection* synapseConnections,
                        ClusterSettings* clusterSettings,
                        const uint32_t* randomValues,
                        const uint32_t neuronBlockPos,
                        const bool isOutputBrick)
{
    __shared__ float tempVal[64][64];

    NeuronBlock* currentNeuronBlock = &neuronBlocks[neuronBlockPos + blockIdx.x];

    // clear temp-values
    for(uint i = 0; i < 64; i++){
        tempVal[threadIdx.x][i] = 0.0f;
    }

    uint32_t synapseBlockId = currentNeuronBlock->backwardNextId;
    while(synapseBlockId != UNINIT_STATE_32)
    {
        // process synapse-sections
        Synapse* section = synapseBlocks[synapseBlockId].synapses[threadIdx.x];
        SynapseConnection* connection = &synapseConnections[synapseBlockId];

        synapseProcessingBackward(section,
                                  connection,
                                  currentNeuronBlock,
                                  neuronBlocks,
                                  clusterSettings,
                                  randomValues,
                                  tempVal[threadIdx.x]);
        __syncthreads();

        if(threadIdx.x < currentNeuronBlock->numberOfNeurons)
        {
            Neuron* neuron = &currentNeuronBlock->neurons[threadIdx.x];
            for(uint i = 0; i < 64; i++)
            {
                neuron->input += tempVal[i][threadIdx.x];
                tempVal[i][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();

        synapseBlockId = connection->backwardNextId;
    }

    if(isOutputBrick == false)
    {
        if(threadIdx.x < currentNeuronBlock->numberOfNeurons)
        {
            Neuron* neuron = &currentNeuronBlock->neurons[threadIdx.x];

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
            neuron->isNew = neuron->active != 0 && neuron->target.blockId == UNINIT_STATE_32;
            neuron->newOffset = 0.0f;
        }
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief run backpropagation for a single synapse-section
 */
__device__ __forceinline__ void
backpropagateSection(Synapse* section,
                     SynapseConnection* connection,
                     const float outH,
                     NeuronBlock* targetNeuronSection,
                     SynapseConnection* synapseConnections,
                     float* tempVal)
{
    float trainValue = 0.2f;
    float counter = outH - connection->offset[threadIdx.x];

    // iterate over all synapses in the section
    for(uint32_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++)
    {
        // break look, if no more synapses to process
        Synapse* synapse = &section[pos];

        if(counter > 0.01f
                && synapse->targetNeuronId != UNINIT_STATE_16)
        {
            // update weight
            trainValue = (float)(126 - synapse->activeCounter) * 0.0002f;
            trainValue += 0.05f;
            Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
            tempVal[threadIdx.x] += targetNeuron->delta * synapse->weight;

            synapse->weight -= trainValue * targetNeuron->delta;
        }

        counter -= synapse->border;
    }
}

//==================================================================================================

/**
 * @brief correct weight of synapses within a segment
 */
__global__ void
reweightCoreSegmentKernel(NeuronBlock* neuronBlocks,
                           SynapseBlock* synapseBlocks,
                           SynapseConnection* synapseConnections,
                          ClusterSettings* clusterSettings,
                          const uint32_t neuronSectionPos)
{
    __shared__ float tempVal[64];

    NeuronBlock* currentNeuronBlock = &neuronBlocks[neuronSectionPos + blockIdx.x];
    tempVal[threadIdx.x] = 0.0f;

    if(threadIdx.x < currentNeuronBlock->numberOfNeurons)
    {
        Neuron* sourceNeuron = &currentNeuronBlock->neurons[threadIdx.x];
        if(sourceNeuron->target.blockId != UNINIT_STATE_32)
        {
            sourceNeuron->delta = 0.0f;
            if(sourceNeuron->active)
            {
                LocationPtr nextId = sourceNeuron->target;

                while(nextId.blockId != UNINIT_STATE_32)
                {
                    Synapse* section = synapseBlocks[nextId.blockId].synapses[nextId.sectionId];
                    SynapseConnection* connection = &synapseConnections[nextId.blockId];
                    NeuronBlock* targetNeuronSection = &neuronBlocks[connection->targetNeuronBlockId];

                    tempVal[threadIdx.x] = sourceNeuron->delta;
                    backpropagateSection(section,
                                         connection,
                                         sourceNeuron->potential,
                                         targetNeuronSection,
                                         synapseConnections,
                                         tempVal);

                    nextId = connection->next[nextId.sectionId];
                }

                tempVal[threadIdx.x] *= 1.4427f * pow(0.5f, sourceNeuron->potential);
                sourceNeuron->delta = tempVal[threadIdx.x];
            }
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
                const uint32_t numberOfSynapseBlocks)
{
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick) {
            processNeuronsOfInputBrickBackward(brick, inputValues, neuronBlocks);
        }
    }

    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick == false)
        {
            prcessCoreSegmentKernel<<<brick->numberOfNeuronBlocks, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->synapseConnections,
                gpuPointer->clusterSettings,
                gpuPointer->randomValues,
                brick->brickBlockPos,
                brick->isOutputBrick);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);

    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
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
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
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

    for(int32_t pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        Brick* brick = &bricks[brickOrder[pos]];              
        reweightCoreSegmentKernel<<<brick->numberOfNeuronBlocks, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->synapseConnections,
                gpuPointer->clusterSettings,
                brick->brickBlockPos);
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

