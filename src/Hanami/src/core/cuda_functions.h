/**
 * @file        cuda_functions.h
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

#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <core/processing/objects.h>

extern "C" void copyToDevice_CUDA(CudaPointerHandle* gpuPointer,
                                  ClusterSettings* clusterSettings,
                                  NeuronBlock* neuronBlocks,
                                  TempNeuronBlock* tempNeuronBlocks,
                                  const uint32_t numberOfNeuronBlocks,
                                  SynapseBlock* synapseBlocks,
                                  const uint32_t numberOfSynapseBlocks,
                                  Brick*,
                                  const uint64_t numberOfBricks);

extern "C" void removeFromDevice_CUDA(CudaPointerHandle* gpuPointer);

extern "C" void copyFromGpu_CUDA(CudaPointerHandle* gpuPointer,
                                 NeuronBlock* neuronBlocks,
                                 const uint32_t numberOfNeuronBlocks,
                                 SynapseBlock* synapseBlocks,
                                 const uint32_t numberOfSynapseBlocks);

extern "C" void processing_CUDA(CudaPointerHandle* gpuPointer,
                                Brick* bricks,
                                const uint64_t numberOfBricks,
                                NeuronBlock* neuronBlocks,
                                const uint32_t numberOfNeuronBlocks,
                                const bool doTrain);

extern "C" void backpropagation_CUDA(CudaPointerHandle* gpuPointer,
                                     Brick* bricks,
                                     const uint32_t uint64_t,
                                     NeuronBlock* neuronBlocks,
                                     TempNeuronBlock* tempNeuronBlocks,
                                     const uint32_t numberOfNeuronBlocks);

extern "C" void update_CUDA(CudaPointerHandle* gpuPointer,
                            NeuronBlock* neuronBlocks,
                            const uint32_t numberOfNeuronBlocks,
                            Brick* bricks,
                            const uint64_t numberOfBricks);

extern "C" void reduction_CUDA(CudaPointerHandle* gpuPointer,
                               Brick* bricks,
                               const uint64_t numberOfBricks,
                               NeuronBlock* neuronBlocks,
                               const uint32_t numberOfNeuronBlocks);

extern "C" uint32_t getNumberOfDevices_CUDA();

extern "C" uint64_t getAvailableMemory_CUDA(const uint32_t deviceId);

#endif  // CUDA_FUNCTIONS_H
