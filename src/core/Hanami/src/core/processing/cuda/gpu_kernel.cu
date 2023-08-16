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

#include <core/processing/objects.h>
#include "../brick.h"

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief initialize a new specific synapse
 */
__device__ __forceinline__ void
createNewSynapse(SynapseConnection* connection,
                 Synapse* synapse,
                 const NeuronSection* targetNeuronSection,
                 const SegmentSettings* segmentSettings,
                 const float outH,
                 const uint* randomValues)
{
    const float maxWeight = outH / (float)(10.0f);

    // set activation-border
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->border = maxWeight * ((float)(randomValues[connection->randomPos]) / (float)(RAND_MAX));

    // set target neuron
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = (ushort)(randomValues[connection->randomPos]
                              % targetNeuronSection->numberOfNeurons);


    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = ((float)(randomValues[connection->randomPos]) / (float)(RAND_MAX)) / 10.0f;

    // update weight with sign
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    const uint signRand = randomValues[connection->randomPos] % 1000;
    synapse->weight *= (float)(1.0f - (1000.0f * segmentSettings->signNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

//==================================================================================================

/**
 * @brief process synapse-section
 */
__device__ __forceinline__ void
synapseProcessingBackward(SynapseSection* section,
                          SynapseConnection* connection,
                          NeuronSection* targetNeuronSection,
                          NeuronSection* neuronSections,
                          NeuronConnection* neuronConnections,
                          SegmentSettings* segmentSettings,
                          const uint* randomValues,
                          float* localMem)
{
    NeuronSection* sourceNeuronSection = &neuronSections[connection->sourceNeuronSectionId];
    Neuron* sourceNeuron = &sourceNeuronSection->neurons[connection->sourceNeuronId];
    const float sourcePotential = sourceNeuron->potential;

    float counter = connection->offset;
    uint pos = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && sourcePotential > counter)
    {
        Synapse* synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(connection,
                             synapse,
                             targetNeuronSection,
                             segmentSettings,
                             sourcePotential,
                             randomValues);
        }

        // update target-neuron
        Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        //targetNeuron->input += synapse->weight;
        localMem[synapse->targetNeuronId] += synapse->weight;

        // update active-counter
        const uint8_t active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * (uint8_t)(synapse->activeCounter < 126);

        // update loop-counter
        counter += synapse->border;
        pos++;
    }

    NeuronConnection* updateSection = &neuronConnections[connection->sourceNeuronSectionId];
    UpdatePos* updatePos = &updateSection->positions[connection->sourceNeuronId];
    updatePos->type = sourcePotential - counter > 0.01f && connection->forwardNextId == UNINIT_STATE_32;
    updatePos->offset = counter + connection->offset;
}

//==================================================================================================

__device__ __forceinline__ void
prcessNeuronConnection(const uint neuronSectionId,
                       NeuronSection* targetNeuronSection,
                       NeuronConnection* neuronConnections,
                       NeuronSection* neuronSections,
                       SynapseConnection* synapseConnections,
                       SynapseSection* synapseSections,
                       SegmentSettings* segmentSettings,
                       const uint* randomValues,
                       float* localMem)
{
    for(uint sectionPos = threadIdx.x;
        sectionPos < NEURON_CONNECTIONS;
        sectionPos += blockDim.x)
    {
        // process synapse-sections
        const uint offset = threadIdx.x * NEURONS_PER_NEURONSECTION;
        const uint sectionId = neuronConnections[neuronSectionId].backwardIds[sectionPos];
        if(sectionId != UNINIT_STATE_32)
        {
            synapseProcessingBackward(&synapseSections[sectionId],
                                      &synapseConnections[sectionId],
                                      targetNeuronSection,
                                      neuronSections,
                                      neuronConnections,
                                      segmentSettings,
                                      randomValues,
                                      &localMem[offset]);
        }

    }
    __syncthreads();

    for(uint sectionPos = threadIdx.x;
        sectionPos < NEURON_CONNECTIONS;
        sectionPos += blockDim.x)
    {
        // apply values of the local-memory to the neurons
        if(threadIdx.x < targetNeuronSection->numberOfNeurons)
        {
            Neuron* neuron = &targetNeuronSection->neurons[threadIdx.x];
            for(uint i = threadIdx.x;
                i < NEURONS_PER_NEURONSECTION * blockDim.x;
                i += NEURONS_PER_NEURONSECTION)
            {
                neuron->input += localMem[i];
                localMem[i] = 0.0f;
            }
        }
    }

    __syncthreads();
}

//==================================================================================================

/**
 * @brief process all neurons within a segment
 */
__global__ void
prcessCoreSegmentKernel(NeuronConnection* neuronConnections,
                        NeuronSection* neuronSections,
                        SynapseConnection* synapseConnections,
                        SynapseSection* synapseSections,
                        SegmentSettings* segmentSettings,
                        float* inputTransfers,
                        float* outputTransfers,
                        const uint32_t* randomValues,
                        const uint32_t neuronSectionPos)
{
    __shared__ uint8_t localMem[4096 * sizeof(float)];
    float* localValues = (float*)&localMem[0];

    const uint32_t neuronSectionId = neuronSectionPos + blockIdx.x;
    NeuronSection* neuronSection = &neuronSections[neuronSectionId];

    prcessNeuronConnection(neuronSectionId,
                           neuronSection,
                           neuronConnections,
                           neuronSections,
                           synapseConnections,
                           synapseSections,
                           segmentSettings,
                           randomValues,
                           localValues);

    if(threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* neuron = &neuronSection->neurons[threadIdx.x];

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

        // handle active-state
        const bool needUpdate = neuron->active != 0 && neuron->targetSectionId == UNINIT_STATE_32;
        UpdatePos* updatePos = &neuronConnections[neuronSectionId].positions[threadIdx.x];
        updatePos->type = needUpdate;
        updatePos->offset = 0.0f;
    }
}

//==================================================================================================

__global__ void
prcessOutputKernel(NeuronConnection* neuronConnections,
                   NeuronSection* neuronSections,
                   SynapseConnection* synapseConnections,
                   SynapseSection* synapseSections,
                   SegmentSettings* segmentSettings,
                   float* outputTransfers,
                   const uint* randomValues,
                   const uint32_t neuronSectionPos)
{
    __shared__ uint8_t localMem[4096 * sizeof(float)];
    float* localValues = (float*)&localMem[0];

    const uint32_t neuronSectionId = neuronSectionPos + blockIdx.x;
    NeuronSection* neuronSection = &neuronSections[neuronSectionId];
    prcessNeuronConnection(neuronSectionId,
                           neuronSection,
                           neuronConnections,
                           neuronSections,
                           synapseConnections,
                           synapseSections,
                           segmentSettings,
                           randomValues,
                           localValues);

    if(threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* neuron = &neuronSection->neurons[threadIdx.x];

        neuron->potential = segmentSettings->potentialOverflow * neuron->input;
        outputTransfers[neuron->targetBorderId] = neuron->potential;
        neuron->input = 0.0f;
    }
}

//==================================================================================================

__global__ void
prcessInputKernel(NeuronSection* neuronSections,
                  NeuronConnection* neuronConnections,
                  float* inputTransfers,
                  const uint32_t neuronSectionPos)
{
    NeuronSection* neuronSection = &neuronSections[neuronSectionPos + blockIdx.x];

    if(threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* neuron = &neuronSection->neurons[threadIdx.x];
        neuron->potential = inputTransfers[neuron->targetBorderId];
        neuron->active = neuron->potential > 0.0f;

        // handle active-state
        const bool needUpdate = neuron->active != 0 && neuron->targetSectionId == UNINIT_STATE_32;
        UpdatePos* updatePos = &neuronConnections[blockIdx.x].positions[threadIdx.x];
        updatePos->type = needUpdate;
        updatePos->offset = 0.0f;
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief run backpropagation for a single synapse-section
 */
__device__ __forceinline__ uint
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     const float outH,
                     NeuronSection* neuronSections,
                     SynapseConnection* synapseConnections,
                     SynapseSection* synapseSections,
                     float* tempVal)
{
    NeuronSection* targetNeuronSection = &neuronSections[connection->targetNeuronSectionId];
    float learnValue = 0.2f;
    float counter = connection->offset;

    // iterate over all synapses in the section
    for(uint32_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++)
    {
        // break look, if no more synapses to process
        Synapse* synapse = &section->synapses[pos];

        if(outH > counter)
        {
            // update weight
            learnValue = (float)(126 - synapse->activeCounter) * 0.0002f;
            learnValue += 0.05f;
            Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
            tempVal[threadIdx.x] += targetNeuron->delta * synapse->weight;

            synapse->weight -= learnValue * targetNeuron->delta;
        }

        counter += synapse->border;
    }

    return connection->forwardNextId;
}

//==================================================================================================

/**
 * @brief correct weight of synapses within a segment
 */
__global__ void
reweightCoreSegmentKernel(NeuronSection* neuronSections,
                          SynapseConnection* synapseConnections,
                          SynapseSection* synapseSections,
                          SegmentSettings* segmentSettings,
                          float* inputTransfers,
                          float* outputTransfers,
                          const uint32_t neuronSectionPos,
                          const bool isInputBrick)
{
    __shared__ float tempVal[64];

    const uint32_t neuronSectionId = neuronSectionPos + blockIdx.x;
    NeuronSection* neuronSection = &neuronSections[neuronSectionId];

    if(threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* sourceNeuron = &neuronSection->neurons[threadIdx.x];
        if(sourceNeuron->targetSectionId != UNINIT_STATE_32)
        {
            sourceNeuron->delta = 0.0f;
            if(sourceNeuron->active)
            {
                uint nextId = sourceNeuron->targetSectionId;
                while(nextId != UNINIT_STATE_32)
                {
                    tempVal[threadIdx.x] = sourceNeuron->delta;
                    nextId = backpropagateSection(&synapseSections[nextId],
                                                  &synapseConnections[nextId],
                                                  sourceNeuron->potential,
                                                  neuronSections,
                                                  synapseConnections,
                                                  synapseSections,
                                                  tempVal);
                }

                tempVal[threadIdx.x] *= 1.4427f * pow(0.5f, sourceNeuron->potential);
                sourceNeuron->delta = tempVal[threadIdx.x];
            }

            if(isInputBrick) {
                outputTransfers[sourceNeuron->targetBorderId] = tempVal[threadIdx.x];
            }
        }
    }
}

//==================================================================================================

__global__ void
reweightOutputKernel(NeuronSection* neuronSections,
                     float* inputTransfers,
                     const uint32_t neuronSectionPos)
{
    NeuronSection* neuronSection = &neuronSections[neuronSectionPos + blockIdx.x];

    if(threadIdx.x < neuronSection->numberOfNeurons)
    {
        neuronSection->neurons[threadIdx.x].delta = inputTransfers[neuronSection->neurons[threadIdx.x].targetBorderId];
        inputTransfers[neuronSection->neurons[threadIdx.x].targetBorderId] = 0.0f;
    }
}

struct PointerHandler
{
    NeuronSection* neuronSections = nullptr;
    SynapseSection* synapseSections = nullptr;
    SegmentSettings* segmentSettings = nullptr;
    float* inputTransfers = nullptr;
    float* outputTransfers = nullptr;
    uint32_t* randomValues = nullptr;
    NeuronConnection* neuronConnections = nullptr;
    SynapseConnection* synapseConnections = nullptr;
};

extern "C"
void
copyToDevice_CUDA(PointerHandler* gpuPointer,
                  SegmentSizes* segmentHeader,
                  SegmentSettings* segmentSettings,
                  NeuronSection* neuronSections,
                  SynapseSection* synapseSections,
                  SynapseConnection* synapseConnections,
                  NeuronConnection* neuronConnections,
                  float* inputTransfers,
                  float* outputTransfers,
                  uint32_t* randomValues)
{
    cudaMalloc(&gpuPointer->neuronSections,     segmentHeader->numberOfNeuronSections     * sizeof(NeuronSection));
    cudaMalloc(&gpuPointer->synapseSections,    segmentHeader->numberOfSynapseSections    * sizeof(SynapseSection));
    cudaMalloc(&gpuPointer->segmentSettings,    1                                         * sizeof(SegmentSettings));
    cudaMalloc(&gpuPointer->inputTransfers,     segmentHeader->numberOfInputTransfers     * sizeof(float));
    cudaMalloc(&gpuPointer->outputTransfers,    segmentHeader->numberOfOutputTransfers    * sizeof(float));
    cudaMalloc(&gpuPointer->randomValues,       NUMBER_OF_RAND_VALUES                     * sizeof(uint32_t));
    cudaMalloc(&gpuPointer->neuronConnections,  segmentHeader->numberOfNeuronSections     * sizeof(NeuronConnection));
    cudaMalloc(&gpuPointer->synapseConnections, segmentHeader->numberOfSynapseSections    * sizeof(SynapseConnection));

    cudaMemcpy(gpuPointer->neuronSections,     neuronSections,     segmentHeader->numberOfNeuronSections    * sizeof(NeuronSection),     cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseSections,    synapseSections,    segmentHeader->numberOfSynapseSections   * sizeof(SynapseSection),    cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->segmentSettings,    segmentSettings,    1                                        * sizeof(SegmentSettings),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->inputTransfers,     inputTransfers,     segmentHeader->numberOfInputTransfers    * sizeof(float),             cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->outputTransfers,    outputTransfers,    segmentHeader->numberOfOutputTransfers   * sizeof(float),             cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->randomValues,       randomValues,       NUMBER_OF_RAND_VALUES                    * sizeof(uint32_t),          cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronConnections,  neuronConnections,  segmentHeader->numberOfNeuronSections    * sizeof(NeuronConnection),  cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseConnections, synapseConnections, segmentHeader->numberOfSynapseSections   * sizeof(SynapseConnection), cudaMemcpyHostToDevice);
}


extern "C"
void
processing_CUDA(PointerHandler* gpuPointer,
                SegmentSizes* segmentHeader,
                uint32_t* brickOrder,
                Brick* bricks,
                float* inputTransfers,
                float* outputTransfers,
                const uint32_t numberOfNeuronSections)
{
    cudaMemcpy(gpuPointer->inputTransfers,
               inputTransfers,
               segmentHeader->numberOfInputTransfers * sizeof(float),
               cudaMemcpyHostToDevice);

    for(uint32_t pos = 0; pos < segmentHeader->numberOfBricks; pos++)
    {
        Brick* brick = &bricks[pos];
        if(brick->isInputBrick == true)
        {
            prcessInputKernel<<<brick->numberOfNeuronSections, NEURONS_PER_NEURONSECTION>>>(
                gpuPointer->neuronSections,
                gpuPointer->neuronConnections,
                gpuPointer->inputTransfers,
                brick->brickBlockPos);
        }
    }

    for(uint32_t pos = 0; pos < segmentHeader->numberOfBricks; pos++)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick == false
                && brick->isOutputBrick == false)
        {
            prcessCoreSegmentKernel<<<brick->numberOfNeuronSections, 64>>>(
                gpuPointer->neuronConnections,
                gpuPointer->neuronSections,
                gpuPointer->synapseConnections,
                gpuPointer->synapseSections,
                gpuPointer->segmentSettings,
                gpuPointer->inputTransfers,
                gpuPointer->outputTransfers,
                gpuPointer->randomValues,
                brick->brickBlockPos);
        }
    }

    for(uint32_t pos = 0; pos < segmentHeader->numberOfBricks; pos++)
    {
        Brick* brick = &bricks[pos];
        if(brick->isOutputBrick == true)
        {
            prcessOutputKernel<<<brick->numberOfNeuronSections, 64>>>(
                gpuPointer->neuronConnections,
                gpuPointer->neuronSections,
                gpuPointer->synapseConnections,
                gpuPointer->synapseSections,
                gpuPointer->segmentSettings,
                gpuPointer->outputTransfers,
                gpuPointer->randomValues,
                brick->brickBlockPos);
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(outputTransfers,
               gpuPointer->outputTransfers,
               segmentHeader->numberOfOutputTransfers * sizeof(float),
               cudaMemcpyDeviceToHost);
}

extern "C"
void
backpropagation_CUDA(PointerHandler* gpuPointer,
                     SegmentSizes* segmentHeader,
                     uint32_t* brickOrder,
                     Brick* bricks,
                     float* inputTransfers,
                     float* outputTransfers,
                     NeuronConnection* neuronConnections,
                     const uint32_t numberOfNeuronSections)
{
    cudaMemcpy(gpuPointer->inputTransfers,
               inputTransfers,
               segmentHeader->numberOfInputTransfers * sizeof(float),
               cudaMemcpyHostToDevice);

    for(int32_t pos = segmentHeader->numberOfBricks - 1; pos >= 0; pos--)
    {
        Brick* brick = &bricks[pos];
        if(brick->isOutputBrick == true)
        {
            reweightOutputKernel<<<brick->numberOfNeuronSections, NEURONS_PER_NEURONSECTION>>> (
                gpuPointer->neuronSections,
                gpuPointer->inputTransfers,
                brick->brickBlockPos);
        }
    }

    for(int32_t pos = segmentHeader->numberOfBricks - 1; pos >= 0; pos--)
    {
        Brick* brick = &bricks[brickOrder[pos]];
        reweightCoreSegmentKernel<<<brick->numberOfNeuronSections, 64>>>(
            gpuPointer->neuronSections,
            gpuPointer->synapseConnections,
            gpuPointer->synapseSections,
            gpuPointer->segmentSettings,
            gpuPointer->inputTransfers,
            gpuPointer->outputTransfers,
            brick->brickBlockPos,
            brick->isInputBrick);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(outputTransfers,
               gpuPointer->outputTransfers,
               segmentHeader->numberOfOutputTransfers * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(neuronConnections,
               gpuPointer->neuronConnections,
               segmentHeader->numberOfNeuronConnections * sizeof(NeuronConnection),
               cudaMemcpyDeviceToHost);
}

extern "C"
void
update_CUDA(PointerHandler* gpuPointer,
            SegmentSizes* segmentHeader,
            NeuronSection* neuronSections,
            SynapseConnection* synapseConnections,
            NeuronConnection* neuronConnections)
{
    cudaMemcpy(gpuPointer->neuronSections,
               neuronSections,
               segmentHeader->numberOfNeuronSections * sizeof(NeuronSection),
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpuPointer->synapseConnections,
               synapseConnections,
               segmentHeader->numberOfSynapseSections * sizeof(SynapseConnection),
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpuPointer->neuronConnections,
               neuronConnections,
               segmentHeader->numberOfNeuronSections * sizeof(NeuronConnection),
               cudaMemcpyHostToDevice);
}
