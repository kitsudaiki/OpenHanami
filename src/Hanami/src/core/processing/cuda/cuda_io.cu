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
 * @param bricks pointer to bricks to initialize their connection-blocks, if exist
 * @param numberOfBricks number of bricks in the cluster to init the connection-block-buffer
 */
extern "C"
void
copyToDevice_CUDA(CudaClusterPointer* gpuPointer,
                  ClusterSettings* clusterSettings,
                  const std::vector<Brick> &bricks,
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

    // initialize connection-blocks all all bricks
    gpuPointer->brickPointer.resize(bricks.size());
    for (uint32_t brickId = 0; brickId < bricks.size(); ++brickId) {
        CudaBrickPointer* cudaBrickPointer = &gpuPointer->brickPointer[brickId];
        const Brick* brick = &bricks[brickId];

        if(brick->neuronBlocks.size() > 0) {
            cudaMalloc(&cudaBrickPointer->neuronBlocks,
                       brick->neuronBlocks.size() * sizeof(NeuronBlock));

            cudaMemcpy(cudaBrickPointer->neuronBlocks,
                       &brick->neuronBlocks[0],
                       brick->neuronBlocks.size() * sizeof(NeuronBlock),
                       cudaMemcpyHostToDevice);
        }

        if(brick->connectionBlocks.size() > 0) {
            cudaMalloc(&cudaBrickPointer->connectionBlocks,
                       brick->connectionBlocks.size() * sizeof(ConnectionBlock));
            cudaMemcpy(cudaBrickPointer->connectionBlocks,
                       &brick->connectionBlocks[0],
                       brick->connectionBlocks.size() * sizeof(ConnectionBlock),
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

    for (uint32_t brickId = 0; brickId < gpuPointer->brickPointer.size(); ++brickId)
    {
        CudaBrickPointer* cudaBrickPointer = &gpuPointer->brickPointer[brickId];

        if (cudaBrickPointer->neuronBlocks != nullptr)
        {
            cudaFree(cudaBrickPointer->neuronBlocks);
            cudaBrickPointer->neuronBlocks = nullptr;
        }

        if (cudaBrickPointer->connectionBlocks != nullptr)
        {
            cudaFree(cudaBrickPointer->connectionBlocks);
            cudaBrickPointer->connectionBlocks = nullptr;
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
                 std::vector<Brick> &bricks)
{
    cudaSetDevice(gpuPointer->deviceId);

    for (uint32_t brickId = 0; brickId < gpuPointer->brickPointer.size(); ++brickId)
    {
        CudaBrickPointer* cudaBrickPointer = &gpuPointer->brickPointer[brickId];
        Brick* brick = &bricks[brickId];

        cudaMemcpy(&brick->neuronBlocks[0],
                   cudaBrickPointer->neuronBlocks,
                   brick->neuronBlocks.size() * sizeof(NeuronBlock),
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
 * @param bricks pointer to local bricks to access and update their connection-blocks
 * @param numberOfBricks number of bricks to update
 */
extern "C"
void
update_CUDA(CudaClusterPointer* gpuPointer,
            std::vector<Brick>& bricks,
            const uint64_t brickId)
{
    cudaSetDevice(gpuPointer->deviceId);

    Brick* brick = &bricks[brickId];

    if (brick->wasResized) {
        // free old connection-block-memory on gpu, if exist
        if (gpuPointer->brickPointer[brickId].connectionBlocks != nullptr)
        {
            cudaFree(gpuPointer->brickPointer[brickId].connectionBlocks);
            gpuPointer->brickPointer[brickId].connectionBlocks = nullptr;
        }

        // allocate to resized memory for the connectionblocks on gpu
        cudaMalloc(&gpuPointer->brickPointer[brickId].connectionBlocks,
                   brick->connectionBlocks.size() * sizeof(ConnectionBlock));
    }

    cudaMemcpy(gpuPointer->brickPointer[brickId].connectionBlocks,
               &brick->connectionBlocks[0],
               brick->connectionBlocks.size() * sizeof(ConnectionBlock),
               cudaMemcpyHostToDevice);

    brick->wasResized = false;
}
