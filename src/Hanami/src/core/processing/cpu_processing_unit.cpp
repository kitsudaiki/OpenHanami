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

#include <hanami_root.h>

#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/section_update.h>
#include <core/processing/cluster_queue.h>
#include <core/cluster/cluster.h>

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
                const uint32_t numberOfSynapseBlocks);

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
                     ClusterSettings* settings);

extern "C"
void
update_CUDA(PointerHandler* gpuPointer,
            NeuronBlock* neuronBlocks,
            const uint32_t numberOfNeuronBlocks,
            SynapseConnection* synapseConnections,
            const uint32_t numberOfSynapseConnections);

uint32_t counter = 0;

/**
 * @brief constructor
 */
CpuProcessingUnit::CpuProcessingUnit()
    : Hanami::Thread("CpuProcessingUnit") {}

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

    cluster->clusterSettings->doTrain = 1;
    if(HanamiRoot::useCuda)
    {
        processing_CUDA(&cluster->gpuPointer,
                        cluster->brickOrder,
                        cluster->bricks,
                        cluster->inputValues,
                        cluster->outputValues,
                        cluster->clusterHeader->bricks.count,
                        cluster->neuronBlocks,
                        cluster->numberOfBrickBlocks,
                        cluster->synapseBlocks,
                        cluster->clusterHeader->synapseBlocks.count);
    }
    else
    {
        prcessCoreSegment(*cluster);
    }
    //prcessCoreSegment(*cluster);

    cluster->clusterSettings->doTrain = 0;
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

    if(HanamiRoot::useCuda)
    {
        backpropagation_CUDA(&cluster->gpuPointer,
                             cluster->brickOrder,
                             cluster->bricks,
                             cluster->outputValues,
                             cluster->expectedValues,
                             cluster->clusterHeader->bricks.count,
                             cluster->neuronBlocks,
                             cluster->numberOfBrickBlocks,
                             cluster->clusterSettings);

        if(updateSections(*cluster))
        {
            update_CUDA(&cluster->gpuPointer,
                        cluster->neuronBlocks,
                        cluster->numberOfBrickBlocks,
                        cluster->synapseConnections,
                        cluster->clusterHeader->synapseConnections.count);
        }
    }
    else
    {
        reweightCoreSegment(*cluster);
    }
    //reweightCoreSegment(*cluster);

    //std::cout<<"counter: "<<counter<<std::endl;
    //counter++;

    if(reductionCounter == 100) {
        //reduceNeurons(*seg);
        reductionCounter = 0;
    }
    reductionCounter++;
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
    if(HanamiRoot::useCuda)
    {
        processing_CUDA(&cluster->gpuPointer,
                        cluster->brickOrder,
                        cluster->bricks,
                        cluster->inputValues,
                        cluster->outputValues,
                        cluster->clusterHeader->bricks.count,
                        cluster->neuronBlocks,
                        cluster->numberOfBrickBlocks,
                        cluster->synapseBlocks,
                        cluster->clusterHeader->synapseBlocks.count);
    }
    else
    {
        prcessCoreSegment(*cluster);
    }

    // send output back if a client-connection is set
    if(cluster->msgClient != nullptr) {
         sendClusterOutputMessage(cluster);
    }
    else
    {
        Task* actualTask = cluster->getActualTask();
        const uint64_t cycle = actualTask->actualCycle;
        if(actualTask->type == IMAGE_REQUEST_TASK)
        {
            // TODO: check for cluster-state instead of client
            const uint32_t hightest = getHighestOutput(*cluster);
            Hanami::DataValue* value = actualTask->resultData.get(cycle).getItemContent()->toValue();
            value->setValue(static_cast<long>(hightest));
        }
        else if(actualTask->type == TABLE_REQUEST_TASK)
        {
            float val = 0.0f;
            for(uint64_t i = 0; i < cluster->clusterHeader->outputValues.count; i++)
            {
                Hanami::DataValue* value = actualTask->resultData.get(cycle).getItemContent()->toValue();
                val = value->getFloat() + cluster->outputValues[i];
                value->setValue(val);
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

    while(m_abort == false)
    {
        cluster = ClusterQueue::getInstance()->getClusterFromQueue();
        if(cluster != nullptr)
        {
            // handle type of processing
            if(cluster->mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
                trainSegmentForward(cluster);
            } else if(cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
                trainSegmentBackward(cluster);
            } else {
                processSegment(cluster);
            }
            cluster->updateClusterState();
        }
        else
        {
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

