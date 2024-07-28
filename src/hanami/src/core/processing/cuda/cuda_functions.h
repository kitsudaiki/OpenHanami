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

#include <core/cluster/objects.h>

extern "C" void copyToDevice_CUDA(CudaClusterPointer* gpuPointer,
                                  ClusterSettings* clusterSettings,
                                  const std::vector<Hexagon>& hexagons,
                                  SynapseBlock* synapseBlocks,
                                  const uint32_t numberOfSynapseBlocks);

extern "C" void removeFromDevice_CUDA(CudaClusterPointer* gpuPointer);

extern "C" void copyFromGpu_CUDA(CudaClusterPointer* gpuPointer,
                                 std::vector<Hexagon>& hexagons,
                                 SynapseBlock* synapseBlocks,
                                 const uint32_t numberOfSynapseBlocks);

extern "C" void update_CUDA(CudaClusterPointer* gpuPointer, Hexagon& hexagon);

extern "C" void processing_CUDA(CudaClusterPointer* gpuPointer,
                                std::vector<Hexagon>& hexagons,
                                const bool doTrain);

extern "C" void backpropagation_CUDA(CudaClusterPointer* gpuPointer,
                                     std::vector<Hexagon>& hexagons);

extern "C" void reduction_CUDA(CudaClusterPointer* gpuPointer, std::vector<Hexagon>& hexagons);

#endif  // CUDA_FUNCTIONS_H
