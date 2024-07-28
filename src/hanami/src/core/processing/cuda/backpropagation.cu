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

#include "../../cluster/objects.h"

/**
 * @brief process neuron in backpropagation-steup
 *
 * @param neuronBlocks pointer to neuron-blocks in gpu-memory
 * @param tempNeuronBlocks pointer to temp-buffer of neuron-blocks in gpu-memory
 * @param neuronBlockPos position-offset within the neuron-block-buffer
 */
__global__ void
backpropagateNeurons(NeuronBlock* neuronBlocks)
{
    __shared__ float localDelta[64];

    const uint64_t neuronBlockId = blockIdx.x;
    NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];
    Neuron* targetNeuron = &targetNeuronBlock->neurons[threadIdx.x];

    if (targetNeuron->active) {
        targetNeuron->delta *= 1.4427f * pow(0.5f, targetNeuron->potential);
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
                    Neuron* sourceNeuron)
{
    __shared__ float localDelta[64];
    __shared__ float localTotalDeltas[64];
    __shared__ float localPotential[64];

    // init values
    localPotential[threadIdx.x] = sourceNeuron->potential - connection->lowerBound;
    Synapse* synapse = nullptr;
    Neuron* targetNeuron = nullptr;
    constexpr float trainValue = 0.05f;
    localTotalDeltas[threadIdx.x] = 0.0f;
    float valid = 0.0f;
    uint8_t active = 0;

    // iterate over all synapses in the section
    for (uint16_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++) {
        synapse = &section->synapses[pos];

        if (synapse->targetNeuronId != UNINIT_STATE_8) {
            targetNeuron = &targetBlock->neurons[synapse->targetNeuronId];
            active = (targetNeuron->potential > 0.0f) == (synapse->weight > 0.0f);
            synapse->activeCounter += active * static_cast<uint8_t>(synapse->activeCounter < 10);

            // calculate new delta
            localDelta[threadIdx.x] = targetNeuron->delta * synapse->weight;
            /*if (localPotential[threadIdx.x] < synapse->border) {
                localDelta[threadIdx.x] *= (1.0f / synapse->border) * localPotential[threadIdx.x];
            }*/

            // update values
            valid = (float)(localPotential[threadIdx.x] > 0.01f);
            synapse->weight -= trainValue * targetNeuron->delta * valid;
            localTotalDeltas[threadIdx.x] += localDelta[threadIdx.x] * valid;
            localPotential[threadIdx.x] -= synapse->border;
        }
    }

    sourceNeuron->delta += localTotalDeltas[threadIdx.x];
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
                         SynapseBlock* synapseBlocks,
                         ConnectionBlock* connectionBlocks,
                         const uint32_t dimY)
{
    ConnectionBlock* connectionBlock = &connectionBlocks[blockIdx.x];
    SynapseConnection* connection = &connectionBlock->connections[threadIdx.x];

    if (connection->origin.blockId != UNINIT_STATE_16) {
        SynapseSection* synapseSection = &synapseBlocks[connectionBlock->targetSynapseBlockPos].sections[threadIdx.x];

        NeuronBlock* sourceNeuronBlock = &neuronBlocks[connection->origin.blockId];
        Neuron* sourceNeuron = &sourceNeuronBlock->neurons[connection->origin.neuronId];

        const uint64_t neuronBlockId = (blockIdx.x / dimY);
        NeuronBlock* targetNeuronBlock = &neuronBlocks[neuronBlockId];

        backpropagateNeuron(synapseSection,
                            connection,
                            targetNeuronBlock,
                            sourceNeuron);
    }
}

/**
 * @brief run backpropagaion on all normal- and output-brikcs to update the weights
 *        of the synapses.
 *
 * @param gpuPointer handle with all gpu-pointer of the cluster
 * @param hexagons pointer to local hexagons
 * @param numberOfHexagons number of hexagons
 * @param neuronBlocks pointer to local neuron-blocks
 * @param tempNeuronBlocks pointer to local temp-values of the neuron-blocks
 * @param numberOfNeuronBlocks number of neuron-blocks
 */
extern "C"
void
backpropagation_CUDA(CudaClusterPointer* gpuPointer,
                     std::vector<Hexagon>& hexagons)
{
    cudaSetDevice(gpuPointer->deviceId);

    // process all hexagons on gpu
    for (int32_t hexagonId = hexagons.size() - 1; hexagonId >= 0; --hexagonId)
    {
        Hexagon* hexagon = &hexagons[hexagonId];
        if (hexagon->header.isInputHexagon) {
            continue;
        }

        // copy necessary data from host to gpu
        cudaMemcpy(gpuPointer->hexagonPointer[hexagonId].neuronBlocks,
                   &hexagon->neuronBlocks[0],
                   hexagon->neuronBlocks.size() * sizeof(NeuronBlock),
                   cudaMemcpyHostToDevice);

        backpropagateNeurons<<<hexagon->header.dimX, 64>>>(
                gpuPointer->hexagonPointer[hexagonId].neuronBlocks);

        backpropagateConnections<<<hexagon->header.dimX, 64>>>(
                gpuPointer->hexagonPointer[hexagonId].neuronBlocks,
                gpuPointer->synapseBlocks,
                gpuPointer->hexagonPointer[hexagonId].connectionBlocks,
                42);

        // copy neurons back to host
        cudaMemcpy(&hexagon->neuronBlocks[0],
                   gpuPointer->hexagonPointer[hexagonId].neuronBlocks,
                   hexagon->neuronBlocks.size() * sizeof(NeuronBlock),
                   cudaMemcpyDeviceToHost);
    }
}
