/**
 * @file        gpu_kernel.cu
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

#include <iostream>
#include <chrono>
#include <math.h>

#include <cuda_runtime_api.h>

#include "../objects.h"
#include "../cluster_io_functions.h"

/**
 * @brief function for generating random-values
 *        coming from this website:
 *            https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 *
 * @param input seed for random value
 *
 * @return random value
 */
__device__ __forceinline__
uint32_t pcg_hash(const uint32_t input)
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief initialize a new synpase
 *
 * @param block source-neuron-block, which is only used to hold the randamo-value
 * @param synapse pointer to the synapse, which should be (re-) initialized
 * @param clusterSettings pointer to the cluster-settings
 * @param remainingW new weight for the synapse
 * @param randomValues pointer to the buffer with all randow-values
 */
__device__ __forceinline__ void
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
 * @param synapseSection current synapse-section to process
 * @param connection pointer to the connection-object, which is related to the section
 * @param targetNeuronBlock neuron-block, which is the target for all synapses in the section
 * @param sourceNeuron pointer to source-neuron, which had triggered the section
 * @param originLocation location of the source-neuron to mark updates
 * @param clusterSettings pointer to cluster-settings
 * @param randomValues pointer to the list with all random-values
 * @param localMem pointer to shared-memory, which should be used by the processing thread
 */
template <bool doTrain>
__device__ __forceinline__ void
synapseProcessingBackward(SynapseSection* synapseSection,
                          SynapseConnection* connection,
                          NeuronBlock* targetNeuronBlock,
                          Neuron* sourceNeuron,
                          const SourceLocationPtr originLocation,
                          ClusterSettings* clusterSettings,
                          uint32_t& randomSeed,
                          float* localMem)
{
    __shared__ float localPotential[64];
    localPotential[threadIdx.x] = sourceNeuron->potential - connection->lowerBound;

    float val = 0.0f;
    uint16_t pos = 0;
    Synapse* synapse = nullptr;

    // iterate over all synapses in the section
    while (pos < SYNAPSES_PER_SYNAPSESECTION && localPotential[threadIdx.x] > 0.01f) {
        synapse = &synapseSection->synapses[pos];

        if constexpr (doTrain) {
            // create new synapse if necesarry and training is active
            if (synapse->targetNeuronId == UNINIT_STATE_8) {
                createNewSynapse(targetNeuronBlock,
                                 synapse,
                                 clusterSettings,
                                 localPotential[threadIdx.x],
                                 randomSeed);
            }

            // split synapse, if necessary
            if (synapse->border > 2.0f * localPotential[threadIdx.x]
                    && pos < SYNAPSES_PER_SYNAPSESECTION - 2)
            {
                const float val = synapse->border / 2.0f;
                synapseSection->synapses[pos + 1].border += val;
                synapse->border -= val;
            }
        }

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            // update target-neuron
            val = synapse->weight;
            if (localPotential[threadIdx.x] < synapse->border) {
                val *= ((1.0f / synapse->border) * localPotential[threadIdx.x]);
            }
            localMem[synapse->targetNeuronId] += val;
        }

        // update loop-counter
        localPotential[threadIdx.x] -= synapse->border;
        ++pos;
    }

    // mark source-neuron for updates, if necessary and training is active
    if constexpr (doTrain) {
        sourceNeuron->isNew = localPotential[threadIdx.x] > 0.01f && synapseSection->hasNext == false;
        sourceNeuron->newLowerBound = (sourceNeuron->potential - localPotential[threadIdx.x]) + connection->lowerBound;
        synapseSection->hasNext = synapseSection->hasNext || sourceNeuron->isNew;
    }
}

/**
 * @brief processSynapses
 *
 * @param neuronBlocks pointer to neuron-blocks in gpu-memory
 * @param synapseBlocks pointer to synapse-blocks in gpu-memory
 * @param connectionBlocks pointer to connection-blocks in gpu-memory
 * @param clusterSettings pointer to cluster-settingss in gpu-memory
 * @param randomValues pointer to list with random-values in gpu-memory
 * @param neuronBlockPos position-offset within the neuron-block-buffer
 * @param dimY number of connections-blocks in y-direction
 */
template <bool doTrain>
__global__ void
processSynapses(NeuronBlock* neuronBlocks,
                SynapseBlock* synapseBlocks,
                ConnectionBlock* connectionBlocks,
                ClusterSettings* clusterSettings,
                uint32_t randomeSeed,
                const uint32_t neuronBlockPos,
                const uint32_t dimY)
{
    SynapseBlock* synapseBlock = nullptr;
    randomeSeed += (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint64_t tid = threadIdx.x;
    const uint64_t neuronBlockId = (blockIdx.x / dimY) + neuronBlockPos;

    // init temp-values, one for each thread and each neuron
    __shared__ float tempVal[64][64];
    for (uint8_t i = 0; i < 64; ++i){
        tempVal[tid][i] = 0.0f;
    }

    // process synapses
    ConnectionBlock* connectionBlock = &connectionBlocks[blockIdx.x];
    SynapseConnection* scon = &connectionBlock->connections[tid];

    if (connectionBlock->targetSynapseBlockPos != UNINIT_STATE_64) {
        synapseBlock =  &synapseBlocks[connectionBlock->targetSynapseBlockPos];

        if (scon->origin.blockId != UNINIT_STATE_32) {
            NeuronBlock* sourceNeuronBlock = &neuronBlocks[scon->origin.blockId];
            Neuron* sourceNeuron = &sourceNeuronBlock->neurons[scon->origin.neuronId];

            if (sourceNeuron->active != 0) {
                SynapseSection* synapseSection = &synapseBlock->sections[tid];
                NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];

                synapseProcessingBackward<doTrain>(synapseSection,
                                                   scon,
                                                   targetNeuronBlock,
                                                   sourceNeuron,
                                                   scon->origin,
                                                   clusterSettings,
                                                   randomeSeed,
                                                   tempVal[tid]);
            }
        }
    }

    __syncthreads();

    // fill temp-values of the synapse-block
    if (connectionBlock->targetSynapseBlockPos != UNINIT_STATE_64) {
        for (uint8_t i = 0; i < 64; ++i) {
            synapseBlock->tempValues[tid] += tempVal[i][tid];
        }
    }
}

/**
 * @brief process neurons
 *
 * @param neuronBlocks pointer to neuron-blocks in gpu-memory
 * @param synapseBlocks pointer to synapse-blocks in gpu-memory
 * @param connectionBlocks pointer to connection-blocks in gpu-memory
 * @param clusterSettings pointer to cluster-settings in gpu-memory
 * @param neuronBlockPos position-offset within the neuron-block-buffer
 * @param dimY number of connections-blocks in y-direction
 * @param isOutputBrick true, if current brick is an output-brick
 */
template <bool doTrain>
__global__ void
processNeurons(NeuronBlock* neuronBlocks,
               SynapseBlock* synapseBlocks,
               ConnectionBlock* connectionBlocks,
               ClusterSettings* clusterSettings,
               const uint32_t neuronBlockPos,
               const uint32_t dimY,
               const bool isOutputBrick)
{
    // init shared memory
    __shared__ float localInputs[64];
    localInputs[threadIdx.x] = 0.0f;

    // init global pointers
    const uint64_t neuronBlockId = blockIdx.x + neuronBlockPos;
    NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];
    Neuron* neuron = &targetNeuronBlock->neurons[threadIdx.x];
    ConnectionBlock* connectionBlock = nullptr;
    SynapseBlock* synapseBlock = nullptr;

    // copy input-values of all releaded synpase-blocks into the neurons
    for (int c = blockIdx.x * dimY; c < (blockIdx.x * dimY) + dimY; ++c) {
        connectionBlock = &connectionBlocks[c];
        if (connectionBlock->targetSynapseBlockPos != UNINIT_STATE_64) {
            synapseBlock =  &synapseBlocks[connectionBlock->targetSynapseBlockPos];
            localInputs[threadIdx.x] += synapseBlock->tempValues[threadIdx.x];
            synapseBlock->tempValues[threadIdx.x] = 0.0f;
        }
    }
    neuron->input = localInputs[threadIdx.x];

    // process neuron-content
    if(isOutputBrick == false)
    {
        neuron->potential /= clusterSettings->neuronCooldown;
        neuron->refractoryTime = neuron->refractoryTime >> 1;

        if (neuron->refractoryTime == 0) {
            neuron->potential = clusterSettings->potentialOverflow * neuron->input;
            neuron->refractoryTime = clusterSettings->refractoryTime;
        }

        neuron->potential -= neuron->border;
        neuron->active = neuron->potential > 0.0f;
        neuron->potential = static_cast<float>(neuron->active) * neuron->potential;
        neuron->input = 0.0f;
        neuron->potential = log2(neuron->potential + 1.0f);

        if constexpr (doTrain) {
            neuron->isNew = neuron->active != 0 && neuron->inUse == 0;
            neuron->newLowerBound = 0.0f;
        }
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief process neuron in backpropagation-steup
 *
 * @param neuronBlocks pointer to neuron-blocks in gpu-memory
 * @param tempNeuronBlocks pointer to temp-buffer of neuron-blocks in gpu-memory
 * @param neuronBlockPos position-offset within the neuron-block-buffer
 */
__global__ void
backpropagateNeurons(NeuronBlock* neuronBlocks,
                     TempNeuronBlock* tempNeuronBlocks,
                     const uint32_t neuronBlockPos)
{
    __shared__ float localDelta[64];

    const uint64_t neuronBlockId = blockIdx.x + neuronBlockPos;
    const NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];
    const Neuron* targetNeuron = &targetNeuronBlock->neurons[threadIdx.x];
    TempNeuronBlock* targetTempBlock = &tempNeuronBlocks[neuronBlockId];
    TempNeuron* targetTempNeuron = &targetTempBlock->neurons[threadIdx.x];

    if (targetNeuron->active) {
        // aggregate different delta-values
        localDelta[threadIdx.x] = 0.0f;
        localDelta[threadIdx.x] += targetTempNeuron->delta[0];
        localDelta[threadIdx.x] += targetTempNeuron->delta[1];
        localDelta[threadIdx.x] += targetTempNeuron->delta[2];
        localDelta[threadIdx.x] += targetTempNeuron->delta[3];
        localDelta[threadIdx.x] += targetTempNeuron->delta[4];
        localDelta[threadIdx.x] += targetTempNeuron->delta[5];
        localDelta[threadIdx.x] += targetTempNeuron->delta[6];
        localDelta[threadIdx.x] += targetTempNeuron->delta[7];

        // calculate new delta-value for next iteration
        localDelta[threadIdx.x] *= 1.4427f * pow(0.5f, targetNeuron->potential);
        targetTempNeuron->delta[0] = localDelta[threadIdx.x];
    }
}

/**
 * @brief backpropagate a synapse-section
 *
 * @param section current synapse-section
 * @param connection current connection related to the synapse-section
 * @param targetTempBlock temp-value-block of the target neuron-block
 * @param sourceNeuron source-neuron, which triggered the section
 * @param sourceTempNeuron temp-balue block of the source-neuron
 */
__device__ __forceinline__ void
backpropagateNeuron(SynapseSection* section,
                    SynapseConnection* connection,
                    NeuronBlock* targetBlock,
                    TempNeuronBlock* targetTempBlock,
                    Neuron* sourceNeuron,
                    TempNeuron* sourceTempNeuron)
{
    __shared__ float localDelta[64];
    __shared__ float localTotalDeltas[64];
    __shared__ float localPotential[64];

    // init values
    localPotential[threadIdx.x] = sourceNeuron->potential - connection->lowerBound;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    TempNeuron* targetTempNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    localTotalDeltas[threadIdx.x] = 0.0f;
    float valid = 0.0f;
    uint8_t active = 0;

    // iterate over all synapses in the section
    for (uint16_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            targetTempNeuron = &targetTempBlock->neurons[synapse->targetNeuronId];
            targetNeuron = &targetBlock->neurons[synapse->targetNeuronId];
            active = (targetNeuron->potential > 0.0f) == (synapse->weight > 0.0f);
            synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 10);

            // calculate new delta
            localDelta[threadIdx.x] = targetTempNeuron->delta[0] * synapse->weight;
            /*if (localPotential[threadIdx.x] < synapse->border) {
                localDelta[threadIdx.x] *= (1.0f / synapse->border) * localPotential[threadIdx.x];
            }*/

            // update values
            valid = (float)(localPotential[threadIdx.x] > 0.01f);
            synapse->weight -= trainValue * targetTempNeuron->delta[0] * valid;
            localTotalDeltas[threadIdx.x] += localDelta[threadIdx.x] * valid;
            localPotential[threadIdx.x] -= synapse->border;
        }
    }

    sourceTempNeuron->delta[connection->origin.posInNeuron] = localTotalDeltas[threadIdx.x];
}

/**
 * @brief backpropagate connections
 *
 * @param neuronBlocks pointer to neuron-blocks in gpu-memory
 * @param tempNeuronBlocks pointer to temp-values of the neuron-blocks in gpu-memory
 * @param synapseBlocks pointer to synapse-blocks in gpu-memory
 * @param connectionBlocks pointer to connection-blocks in gpu-memory
 * @param neuronBlockPos position-offset within the neuron-block-buffer
 * @param dimY number of connections-blocks in y-direction
 */
__global__ void
backpropagateConnections(NeuronBlock* neuronBlocks,
                         TempNeuronBlock* tempNeuronBlocks,
                         SynapseBlock* synapseBlocks,
                         ConnectionBlock* connectionBlocks,
                         const uint32_t neuronBlockPos,
                         const uint32_t dimY)
{
    ConnectionBlock* connectionBlock = &connectionBlocks[blockIdx.x];
    SynapseConnection* connection = &connectionBlock->connections[threadIdx.x];

    if (connection->origin.blockId != UNINIT_STATE_32) {
        SynapseSection* synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[threadIdx.x];

        NeuronBlock* sourceNeuronBlock = &neuronBlocks[connection->origin.blockId];
        TempNeuronBlock* sourceTempBlock = &tempNeuronBlocks[connection->origin.blockId];
        Neuron* sourceNeuron = &sourceNeuronBlock->neurons[connection->origin.neuronId];
        TempNeuron* sourceTempNeuron = &sourceTempBlock->neurons[connection->origin.neuronId];

        const uint64_t neuronBlockId = (blockIdx.x / dimY)  + neuronBlockPos;
        TempNeuronBlock* targetTempBlock = &tempNeuronBlocks[neuronBlockId];
        NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];

        backpropagateNeuron(synapseSection,
                            connection,
                            targetNeuronBlock,
                            targetTempBlock,
                            sourceNeuron,
                            sourceTempNeuron);
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief backpropagate a synapse-section
 *
 * @param section current synapse-section
 */
__device__ __forceinline__ bool
reduceSection(SynapseSection* section)
{
    Synapse* synapse;
    uint8_t exist = 0;

    for (uint8_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            synapse->activeCounter -= static_cast<uint8_t>(synapse->activeCounter < 10);

            // handle active-counter
            if (synapse->activeCounter == 0) {
                if (pos < SYNAPSES_PER_SYNAPSESECTION - 1) {
                    section->synapses[pos] = section->synapses[pos + 1];
                    section->synapses[pos + 1] = Synapse();
                } else {
                    section->synapses[pos] = Synapse();
                }
            }
            else {
                exist++;
            }
        }
    }

    // return true;
    return exist != 0;
}

/**
 * @brief reduce synapse, in order to limit the amount of memory
 *
 * @param connectionBlocks pointer to connection-blocks
 * @param neuronBlocks pointer to neuron-blocks
 * @param synapseBlocks pointer to synapse-blocks
 */
__global__ void
reduceConnections(ConnectionBlock* connectionBlocks,
                  NeuronBlock* neuronBlocks,
                  SynapseBlock* synapseBlocks)
{
    Neuron* sourceNeuron = nullptr;
    NeuronBlock* sourceNeuronBlock = nullptr;
    SynapseSection* synapseSection = nullptr;

    ConnectionBlock* connectionBlock = &connectionBlocks[blockIdx.x];
    SynapseConnection* connection = &connectionBlock->connections[threadIdx.x];

    if (connection->origin.blockId != UNINIT_STATE_32) {
        synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[threadIdx.x];
        sourceNeuronBlock = &neuronBlocks[connection->origin.blockId];
        sourceNeuron = &sourceNeuronBlock->neurons[connection->origin.neuronId];

        // if section is complete empty, then erase it
        if (reduceSection(synapseSection) == false) {
            // initialize the creation of a new section
            sourceNeuron->isNew = 1;
            sourceNeuron->newLowerBound = connection->lowerBound;
            sourceNeuron->inUse &= (~(1 << connection->origin.posInNeuron));

            // mark current connection as available again
            connection->origin.blockId = UNINIT_STATE_32;
            connection->origin.neuronId = UNINIT_STATE_16;
            connection->origin.posInNeuron = 0;
        }
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief initial copy of data from the host to the gpu
 *
 * @param gpuPointer pointer to the handle-object, which will store the pointer for the gpu-buffer
 * @param clusterSettings pointer to cluster-settings on host
 * @param neuronBlocks pointer to neuron-blocks on host
 * @param tempNeuronBlocks pointer to temp-values of the neuron-blocks on host
 * @param numberOfNeuronBlocks number of neuron-blocks to copy
 * @param synapseBlocks pointer to synapse-blocks on host
 * @param numberOfSynapseBlocks number of synapse-blocks to copy
 * @param bricks pointer to bricks to initialize their connection-blocks, if exist
 * @param numberOfBricks number of bricks in the cluster to init the connection-block-buffer
 */
extern "C"
void
copyToDevice_CUDA(CudaPointerHandle* gpuPointer,
                  ClusterSettings* clusterSettings,
                  NeuronBlock* neuronBlocks,
                  TempNeuronBlock* tempNeuronBlocks,
                  const uint32_t numberOfNeuronBlocks,
                  SynapseBlock* synapseBlocks,
                  const uint32_t numberOfSynapseBlocks,
                  Brick* bricks,
                  const uint64_t numberOfBricks)
{
    cudaSetDevice(gpuPointer->deviceId);

    // allocate memory on gpu
    cudaMalloc(&gpuPointer->clusterSettings, 1                     * sizeof(ClusterSettings));
    cudaMalloc(&gpuPointer->neuronBlocks,    numberOfNeuronBlocks  * sizeof(NeuronBlock));
    cudaMalloc(&gpuPointer->tempNeuronBlock, numberOfNeuronBlocks  * sizeof(TempNeuronBlock));
    cudaMalloc(&gpuPointer->synapseBlocks,   numberOfSynapseBlocks * sizeof(SynapseBlock));

    // copy data from host into the allocated memory
    cudaMemcpy(gpuPointer->clusterSettings, clusterSettings,  1                     * sizeof(ClusterSettings), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronBlocks,    neuronBlocks,     numberOfNeuronBlocks  * sizeof(NeuronBlock),     cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->tempNeuronBlock, tempNeuronBlocks, numberOfNeuronBlocks  * sizeof(TempNeuronBlock), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseBlocks,   synapseBlocks,    numberOfSynapseBlocks * sizeof(SynapseBlock),    cudaMemcpyHostToDevice);

    // initialize connection-blocks all all bricks
    gpuPointer->connectionBlocks.resize(numberOfBricks);
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        gpuPointer->connectionBlocks[brickId] = nullptr;

        Brick* brick = &bricks[brickId];
        if(brick->connectionBlocks->size() > 0) {
            cudaMalloc(&gpuPointer->connectionBlocks[brickId],
                       brick->connectionBlocks->size() * sizeof(ConnectionBlock));
            cudaMemcpy(gpuPointer->connectionBlocks[brickId],
                       &brick->connectionBlocks[0][0],
                       brick->connectionBlocks->size() * sizeof(ConnectionBlock),
                       cudaMemcpyHostToDevice);
        }
    }
}

/**
 * @brief removed all data from the gpu, which are linked in the handle-object
 *
 * @param gpuPointer handle with all pointer to free
 */
extern "C"
void
removeFromDevice_CUDA(CudaPointerHandle* gpuPointer)
{
    cudaSetDevice(gpuPointer->deviceId);

    for (uint32_t c = 0; c < gpuPointer->connectionBlocks.size(); ++c)
    {
        // free old connection-block-memory on gpu, if exist
        if (gpuPointer->connectionBlocks[c] != nullptr)
        {
            cudaFree(gpuPointer->connectionBlocks[c]);
            gpuPointer->connectionBlocks[c] = nullptr;
        }
    }

    cudaFree(gpuPointer->clusterSettings);
    cudaFree(gpuPointer->neuronBlocks);
    cudaFree(gpuPointer->tempNeuronBlock);
    cudaFree(gpuPointer->synapseBlocks);
}

/**
 * @brief copy all data from the gpu back to the host
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param neuronBlocks pointer to neuron-blocks on host
 * @param numberOfNeuronBlocks number of neuron-blocks to copy
 * @param synapseBlocks pointer to synpase-blocks on host
 * @param numberOfSynapseBlocks number of synpase-blocks to copy
 */
extern "C"
void
copyFromGpu_CUDA(CudaPointerHandle* gpuPointer,
                 NeuronBlock* neuronBlocks,
                 const uint32_t numberOfNeuronBlocks,
                 SynapseBlock* synapseBlocks,
                 const uint32_t numberOfSynapseBlocks)
{
    cudaSetDevice(gpuPointer->deviceId);

    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(synapseBlocks,
               gpuPointer->synapseBlocks,
               numberOfSynapseBlocks * sizeof(SynapseBlock),
               cudaMemcpyDeviceToHost);
}

/**
 * @brief in case the cluster was resized, these changes have to be pushed to the gpu
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param neuronBlocks pointer to local buffer with neuron-blocks to update
 * @param numberOfNeuronBlocks number of neuron-blocks to update
 * @param bricks pointer to local bricks to access and update their connection-blocks
 * @param numberOfBricks number of bricks to update
 */
extern "C"
void
update_CUDA(CudaPointerHandle* gpuPointer,
            NeuronBlock* neuronBlocks,
            const uint32_t numberOfNeuronBlocks,
            Brick* bricks,
            const uint64_t numberOfBricks)
{
    cudaSetDevice(gpuPointer->deviceId);

    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];

        if (brick->wasResized) {
            // free old connection-block-memory on gpu, if exist
            if (gpuPointer->connectionBlocks[brickId] != nullptr)
            {
                cudaFree(gpuPointer->connectionBlocks[brickId]);
                gpuPointer->connectionBlocks[brickId] = nullptr;
            }

            // allocate to resized memory for the connectionblocks on gpu
            cudaMalloc(&gpuPointer->connectionBlocks[brickId],
                       brick->connectionBlocks->size() * sizeof(ConnectionBlock));
        }

        cudaMemcpy(gpuPointer->connectionBlocks[brickId],
                   &brick->connectionBlocks[0][0],
                   brick->connectionBlocks->size() * sizeof(ConnectionBlock),
                   cudaMemcpyHostToDevice);

        brick->wasResized = false;
    }
}

/**
 * @brief process all normal- and output-bricks and train them, if wanted.
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param bricks pointer to local bricks
 * @param numberOfBricks number of bricks
 * @param neuronBlocks pointer to local neuron-block
 * @param numberOfNeuronBlocks number of neuron-blokcs
 * @param doTrain true to run a taining-process
 */
extern "C"
void
processing_CUDA(CudaPointerHandle* gpuPointer,
                Brick* bricks,
                const uint64_t numberOfBricks,
                NeuronBlock* neuronBlocks,
                const uint32_t numberOfNeuronBlocks,
                const bool doTrain)
{
    cudaSetDevice(gpuPointer->deviceId);

    uint32_t randomeSeed = rand();

    // copy necessary data from host to gpu
    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    // process bricks on gpu
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick) {
            continue;
        }

        if (doTrain)
        {
            processSynapses<true><<<brick->dimX * brick->dimY, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                randomeSeed + brickId,
                brick->neuronBlockPos,
                brick->dimY);

            processNeurons<true><<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                brick->neuronBlockPos,
                brick->dimY,
                brick->isOutputBrick);
        }
        else
        {
            processSynapses<false><<<brick->dimX * brick->dimY, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                randomeSeed + brickId,
                brick->neuronBlockPos,
                brick->dimY);

            processNeurons<false><<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->clusterSettings,
                brick->neuronBlockPos,
                brick->dimY,
                brick->isOutputBrick);
        }
    }

    // copy resulting data back to host
    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
}

/**
 * @brief run backpropagaion on all normal- and output-brikcs to update the weights
 *        of the synapses.
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param bricks pointer to local bricks
 * @param numberOfBricks number of bricks
 * @param neuronBlocks pointer to local neuron-blocks
 * @param tempNeuronBlocks pointer to local temp-values of the neuron-blocks
 * @param numberOfNeuronBlocks number of neuron-blocks
 */
extern "C"
void
backpropagation_CUDA(CudaPointerHandle* gpuPointer,
                     Brick* bricks,
                     const uint32_t uint64_t,
                     NeuronBlock* neuronBlocks,
                     TempNeuronBlock* tempNeuronBlocks,
                     const uint32_t numberOfNeuronBlocks)
{
    cudaSetDevice(gpuPointer->deviceId);

    // copy necessary data from host to gpu
    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->tempNeuronBlock,
               tempNeuronBlocks,
               numberOfNeuronBlocks * sizeof(TempNeuronBlock),
               cudaMemcpyHostToDevice);

    // process all bricks on gpu
    for (int32_t brickId = uint64_t - 1; brickId >= 0; --brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick) {
            continue;
        }

        backpropagateNeurons<<<brick->dimX, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->tempNeuronBlock,
                brick->neuronBlockPos);

        backpropagateConnections<<<brick->dimX * brick->dimY, 64>>>(
                gpuPointer->neuronBlocks,
                gpuPointer->tempNeuronBlock,
                gpuPointer->synapseBlocks,
                gpuPointer->connectionBlocks[brickId],
                brick->neuronBlockPos,
                brick->dimY);
    }

    // copy neurons back to host
    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
}

/**
 * @brief run backpropagaion on all normal- and output-brikcs to update the weights
 *        of the synapses.
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param bricks pointer to local bricks
 * @param numberOfBricks number of bricks
 * @param neuronBlocks pointer to local neuron-blocks
 * @param numberOfNeuronBlocks number of neuron-blocks
 */
extern "C"
void
reduction_CUDA(CudaPointerHandle* gpuPointer,
               Brick* bricks,
               const uint64_t numberOfBricks,
               NeuronBlock* neuronBlocks,
               const uint32_t numberOfNeuronBlocks)
{
    cudaSetDevice(gpuPointer->deviceId);

    // copy necessary data from host to gpu
    cudaMemcpy(gpuPointer->neuronBlocks,
               neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyHostToDevice);

    // process all bricks on gpu
    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId)
    {
        Brick* brick = &bricks[brickId];
        if (brick->isInputBrick) {
            continue;
        }

        reduceConnections<<<brick->dimX, 64>>>(
                gpuPointer->connectionBlocks[brickId],
                gpuPointer->neuronBlocks,
                gpuPointer->synapseBlocks);


        cudaMemcpy(&brick->connectionBlocks[0][0],
                   gpuPointer->connectionBlocks[brickId],
                   brick->connectionBlocks->size() * sizeof(ConnectionBlock),
                   cudaMemcpyDeviceToHost);
    }

    // copy neurons back to host
    cudaMemcpy(neuronBlocks,
               gpuPointer->neuronBlocks,
               numberOfNeuronBlocks * sizeof(NeuronBlock),
               cudaMemcpyDeviceToHost);
}

/**
 * @brief get number of available cuda-devices
 */
extern "C"
uint32_t
getNumberOfDevices_CUDA()
{
    int deviceCount;
    cudaError_t cudaResult = cudaGetDeviceCount(&deviceCount);

    if (cudaResult != cudaSuccess || deviceCount == 0) {
        return 0;
    }

    return deviceCount;
}

/**
 * @brief get available memory on device
 *
 * @param deviceId id of the device to check
 *
 * @return size of free memory on device in bytes
 */
extern "C"
uint64_t
getAvailableMemory_CUDA(const uint32_t deviceId)
{
    cudaSetDevice(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    // totalMem;

    return freeMem;
}
