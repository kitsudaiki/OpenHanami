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
 * @brief initial copy of data from the host to the gpu
 *
 * @param gpuPointer pointer to the handle-object, which will store the pointer for the gpu-buffer
 * @param clusterSettings pointer to cluster-settings on host
 * @param neuronBlocks pointer to neuron-blocks on host
 * @param tempNeuronBlocks pointer to temp-values of the neuron-blocks on host
 * @param numberOfNeuronBlocks number of neuron-blocks to copy
 * @param synapseBlocks pointer to synapse-blocks on host
 * @param numberOfSynapseBlocks number of synapse-blocks to copy
 * @param hexagons pointer to hexagons to initialize their connection-blocks, if exist
 * @param numberOfHexagons number of hexagons in the cluster to init the connection-block-buffer
 */
extern "C"
void
copyToDevice_CUDA(CudaClusterPointer* gpuPointer,
                  ClusterSettings* clusterSettings,
                  const std::vector<Hexagon> &hexagons,
                  SynapseBlock* synapseBlocks,
                  const uint32_t numberOfSynapseBlocks)
{
    cudaSetDevice(gpuPointer->deviceId);

    // allocate memory on gpu
    cudaMalloc(&gpuPointer->clusterSettings, 1                     * sizeof(ClusterSettings));
    cudaMalloc(&gpuPointer->synapseBlocks,   numberOfSynapseBlocks * sizeof(SynapseBlock));

    // copy data from host into the allocated memory
    cudaMemcpy(gpuPointer->clusterSettings, clusterSettings,  1                     * sizeof(ClusterSettings), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseBlocks,   synapseBlocks,    numberOfSynapseBlocks * sizeof(SynapseBlock),    cudaMemcpyHostToDevice);

    // initialize connection-blocks all all hexagons
    gpuPointer->hexagonPointer.resize(hexagons.size());
    for (uint32_t hexagonId = 0; hexagonId < hexagons.size(); ++hexagonId) {
        CudaHexagonPointer* cudaHexagonPointer = &gpuPointer->hexagonPointer[hexagonId];
        const Hexagon* hexagon = &hexagons[hexagonId];

        if(hexagon->neuronBlocks.size() > 0) {
            cudaMalloc(&cudaHexagonPointer->neuronBlocks,
                       hexagon->neuronBlocks.size() * sizeof(NeuronBlock));

            cudaMemcpy(cudaHexagonPointer->neuronBlocks,
                       &hexagon->neuronBlocks[0],
                       hexagon->neuronBlocks.size() * sizeof(NeuronBlock),
                       cudaMemcpyHostToDevice);
        }

        if(hexagon->connectionBlocks.size() > 0) {
            cudaMalloc(&cudaHexagonPointer->connectionBlocks,
                       hexagon->connectionBlocks.size() * sizeof(ConnectionBlock));
            cudaMemcpy(cudaHexagonPointer->connectionBlocks,
                       &hexagon->connectionBlocks[0],
                       hexagon->connectionBlocks.size() * sizeof(ConnectionBlock),
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
removeFromDevice_CUDA(CudaClusterPointer* gpuPointer)
{
    cudaSetDevice(gpuPointer->deviceId);

    cudaFree(gpuPointer->clusterSettings);
    cudaFree(gpuPointer->synapseBlocks);

    for (uint32_t hexagonId = 0; hexagonId < gpuPointer->hexagonPointer.size(); ++hexagonId)
    {
        CudaHexagonPointer* cudaHexagonPointer = &gpuPointer->hexagonPointer[hexagonId];

        if (cudaHexagonPointer->neuronBlocks != nullptr)
        {
            cudaFree(cudaHexagonPointer->neuronBlocks);
            cudaHexagonPointer->neuronBlocks = nullptr;
        }

        if (cudaHexagonPointer->connectionBlocks != nullptr)
        {
            cudaFree(cudaHexagonPointer->connectionBlocks);
            cudaHexagonPointer->connectionBlocks = nullptr;
        }
    }
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
copyFromGpu_CUDA(CudaClusterPointer* gpuPointer,
                 SynapseBlock* synapseBlocks,
                 const uint32_t numberOfSynapseBlocks,
                 std::vector<Hexagon> &hexagons)
{
    cudaSetDevice(gpuPointer->deviceId);

    for (uint32_t hexagonId = 0; hexagonId < gpuPointer->hexagonPointer.size(); ++hexagonId)
    {
        CudaHexagonPointer* cudaHexagonPointer = &gpuPointer->hexagonPointer[hexagonId];
        Hexagon* hexagon = &hexagons[hexagonId];

        cudaMemcpy(&hexagon->neuronBlocks[0],
                   cudaHexagonPointer->neuronBlocks,
                   hexagon->neuronBlocks.size() * sizeof(NeuronBlock),
                   cudaMemcpyDeviceToHost);
    }

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
 * @param hexagons pointer to local hexagons to access and update their connection-blocks
 * @param numberOfHexagons number of hexagons to update
 */
extern "C"
void
update_CUDA(CudaClusterPointer* gpuPointer,
            std::vector<Hexagon>& hexagons,
            const uint64_t hexagonId)
{
    cudaSetDevice(gpuPointer->deviceId);

    Hexagon* hexagon = &hexagons[hexagonId];

    if (hexagon->wasResized) {
        // free old connection-block-memory on gpu, if exist
        if (gpuPointer->hexagonPointer[hexagonId].connectionBlocks != nullptr)
        {
            cudaFree(gpuPointer->hexagonPointer[hexagonId].connectionBlocks);
            gpuPointer->hexagonPointer[hexagonId].connectionBlocks = nullptr;
        }

        // allocate to resized memory for the connectionblocks on gpu
        cudaMalloc(&gpuPointer->hexagonPointer[hexagonId].connectionBlocks,
                   hexagon->connectionBlocks.size() * sizeof(ConnectionBlock));
    }

    cudaMemcpy(gpuPointer->hexagonPointer[hexagonId].connectionBlocks,
               &hexagon->connectionBlocks[0],
               hexagon->connectionBlocks.size() * sizeof(ConnectionBlock),
               cudaMemcpyHostToDevice);

    hexagon->wasResized = false;
}
