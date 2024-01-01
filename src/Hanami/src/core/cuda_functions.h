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
                                  const uint32_t numberOfBricks,
                                  uint32_t* randomValues);

extern "C" void removeFromDevice_CUDA(CudaPointerHandle* gpuPointer);

extern "C" void processing_CUDA(CudaPointerHandle* gpuPointer,
                                Brick* bricks,
                                const uint32_t numberOfBricks,
                                NeuronBlock* neuronBlocks,
                                const uint32_t numberOfNeuronBlocks,
                                const bool doTrain);

extern "C" void backpropagation_CUDA(CudaPointerHandle* gpuPointer,
                                     Brick* bricks,
                                     const uint32_t numberOfBricks,
                                     NeuronBlock* neuronBlocks,
                                     TempNeuronBlock* tempNeuronBlocks,
                                     const uint32_t numberOfNeuronBlocks);

extern "C" void update_CUDA(CudaPointerHandle* gpuPointer,
                            NeuronBlock* neuronBlocks,
                            const uint32_t numberOfNeuronBlocks,
                            Brick* bricks,
                            const uint32_t numberOfBricks);

#endif  // CUDA_FUNCTIONS_H
