/**
 * @file        cpu_processing_unit.cpp
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

#include "cpu_processing_unit.h"

#include <core/cluster/cluster.h>
#include <core/processing/cluster_queue.h>
#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/section_update.h>
#include <hanami_root.h>

static uint64_t counter = 0;

extern "C" void processing_CUDA(PointerHandler* gpuPointer,
                                Brick* bricks,
                                float* inputValues,
                                float* outputValues,
                                const uint32_t numberOfBricks,
                                NeuronBlock* neuronBlocks,
                                const uint32_t numberOfNeuronBlocks,
                                const bool doTrain);

extern "C" void backpropagation_CUDA(PointerHandler* gpuPointer,
                                     Brick* bricks,
                                     float* outputValues,
                                     float* expectedValues,
                                     const uint32_t numberOfBricks,
                                     NeuronBlock* neuronBlocks,
                                     TempNeuronBlock* tempNeuronBlocks,
                                     const uint32_t numberOfNeuronBlocks,
                                     ClusterSettings* settings);

extern "C" void update_CUDA(PointerHandler* gpuPointer,
                            NeuronBlock* neuronBlocks,
                            const uint32_t numberOfNeuronBlocks,
                            Brick* bricks,
                            const uint32_t numberOfBricks);

/**
 * @brief constructor
 */
CpuProcessingUnit::CpuProcessingUnit() : Hanami::Thread("CpuProcessingUnit") {}

/**
 * @brief destructor
 */
CpuProcessingUnit::~CpuProcessingUnit() {}

/**
 * @brief run forward-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CpuProcessingUnit::trainSegmentForward(Cluster* cluster)
{
    Hanami::ErrorContainer error;

    if (HanamiRoot::useCuda) {
        processing_CUDA(&cluster->gpuPointer,
                        cluster->bricks,
                        cluster->inputValues,
                        cluster->outputValues,
                        cluster->clusterHeader->bricks.count,
                        cluster->neuronBlocks,
                        cluster->numberOfNeuronBlocks,
                        true);

        if (updateSections(*cluster)) {
            update_CUDA(&cluster->gpuPointer,
                        cluster->neuronBlocks,
                        cluster->numberOfNeuronBlocks,
                        cluster->bricks,
                        cluster->clusterHeader->bricks.count);
        }
        std::cout << "counter: " << counter << std::endl;
        counter++;
    }
    else {
        prcessCoreSegment(*cluster, true);
    }
    // prcessCoreSegment(*cluster);
}

/**
 * @brief run back-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CpuProcessingUnit::trainSegmentBackward(Cluster* cluster)
{
    Hanami::ErrorContainer error;

    if (HanamiRoot::useCuda) {
        backpropagation_CUDA(&cluster->gpuPointer,
                             cluster->bricks,
                             cluster->outputValues,
                             cluster->expectedValues,
                             cluster->clusterHeader->bricks.count,
                             cluster->neuronBlocks,
                             cluster->tempNeuronBlocks,
                             cluster->numberOfNeuronBlocks,
                             &cluster->clusterHeader->settings);
    }
    else {
        reweightCoreSegment(*cluster);
    }
    // reweightCoreSegment(*cluster);

    if (reductionCounter == 100) {
        // reduceNeurons(*seg);
        reductionCounter = 0;
    }
    reductionCounter++;
}

/**
 * @brief get position of the highest output-position
 *
 * @param cluster output-cluster to check
 *
 * @return position of the highest output.
 */
uint32_t
getHighestOutput(const Cluster& cluster)
{
    float hightest = -0.1f;
    uint32_t hightestPos = 0;
    float value = 0.0f;

    for (uint32_t outputNeuronId = 0; outputNeuronId < cluster.clusterHeader->numberOfOutputs;
         outputNeuronId++)
    {
        value = cluster.outputValues[outputNeuronId];
        if (value > hightest) {
            hightest = value;
            hightestPos = outputNeuronId;
        }
    }

    return hightestPos;
}

/**
 * @brief process segments
 *
 * @param cluster cluster to process
 */
void
CpuProcessingUnit::processSegment(Cluster* cluster)
{
    Hanami::ErrorContainer error;
    if (HanamiRoot::useCuda) {
        processing_CUDA(&cluster->gpuPointer,
                        cluster->bricks,
                        cluster->inputValues,
                        cluster->outputValues,
                        cluster->clusterHeader->bricks.count,
                        cluster->neuronBlocks,
                        cluster->numberOfNeuronBlocks,
                        false);
    }
    else {
        prcessCoreSegment(*cluster, false);
    }

    // send output back if a client-connection is set
    if (cluster->msgClient != nullptr) {
        sendClusterOutputMessage(cluster);
    }
    else {
        Task* actualTask = cluster->getActualTask();
        const uint64_t cycle = actualTask->actualCycle;
        if (actualTask->type == IMAGE_REQUEST_TASK) {
            // TODO: check for cluster-state instead of client
            const uint32_t hightest = getHighestOutput(*cluster);
            actualTask->resultData[cycle] = static_cast<long>(hightest);
        }
        else if (actualTask->type == TABLE_REQUEST_TASK) {
            float val = 0.0f;
            for (uint64_t i = 0; i < cluster->clusterHeader->numberOfOutputs; i++) {
                const float temp = actualTask->resultData[cycle];
                val = temp + cluster->outputValues[i];
                actualTask->resultData[cycle] = val;
            }
        }
    }
}

/**
 * @brief run loop to process all available segments
 */
void
CpuProcessingUnit::run()
{
    Cluster* cluster = nullptr;

    while (m_abort == false) {
        cluster = ClusterQueue::getInstance()->getClusterFromQueue();
        if (cluster != nullptr) {
            // handle type of processing
            if (cluster->mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
                trainSegmentForward(cluster);
            }
            else if (cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
                trainSegmentBackward(cluster);
            }
            else {
                processSegment(cluster);
            }
            cluster->updateClusterState();
        }
        else {
            // if no segments are available then sleep
            sleepThread(1000);
        }
    }
}

/**
 * @brief SingleThreadProcessingStatic::reductionTraining

void
CpuProcessingUnit::reductionTraining(DynamicSegment* synapseSegment)
{
    const float initError = calculateSegmentError(synapseSegment);
    float error = initError;

    if(initError > 0.1f)
    {
        int16_t timeout = 10;
        while(error >= initError
              && timeout >= 0)
        {
            reduceSegment(synapseSegment);
            execute(synapseSegment);
            error = calculateSegmentError(synapseSegment);

            timeout--;
        }
    }
}*/
