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

#include <core/segments/core_segment/core_segment.h>
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>

#include <core/cluster/cluster.h>
#include <core/processing/segment_queue.h>

#include <kyouko_root.h>

#include <core/segments/core_segment/backpropagation.h>
#include <core/segments/core_segment/processing.h>
#include <core/segments/core_segment/reduction.h>
#include <core/segments/core_segment/section_update.h>

#include <core/segments/output_segment/backpropagation.h>
#include <core/segments/output_segment/processing.h>

#include <core/segments/input_segment/processing.h>

#include <libKitsunemimiOpencl/gpu_interface.h>
#include <libKitsunemimiOpencl/gpu_handler.h>

extern "C"
void
processing_CUDA(PointerHandler* gpuPointer,
                SegmentSizes* segmentHeader,
                uint32_t* brickOrder,
                BrickHeader* bricks,
                float* inputTransfers,
                float* outputTransfers,
                const uint32_t numberOfNeuronSections);

extern "C"
void
backpropagation_CUDA(PointerHandler* gpuPointer,
                     SegmentSizes* segmentHeader,
                     uint32_t* brickOrder,
                     BrickHeader* bricks,
                     float* inputTransfers,
                     float* outputTransfers,
                     UpdatePosSection* updatePosSections,
                     const uint32_t numberOfNeuronSections);

extern "C"
void
update_CUDA(PointerHandler* gpuPointer,
            SegmentSizes* segmentHeader,
            UpdatePosSection* updatePosSections,
            NeuronSection* neuronSections,
            SynapseConnection* synapseConnections,
            NeuronConnection* neuronConnections);

uint32_t counter = 0;

/**
 * @brief constructor
 */
CpuProcessingUnit::CpuProcessingUnit()
    : Kitsunemimi::Thread("CpuProcessingUnit") {}

/**
 * @brief destructor
 */
CpuProcessingUnit::~CpuProcessingUnit() {}

/**
 * @brief run forward-propagation on a segment
 *
 * @param segment segment to process
 */
void
CpuProcessingUnit::learnSegmentForward(AbstractSegment* segment)
{
    Kitsunemimi::ErrorContainer error;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    switch(segment->getType())
    {
        case CORE_SEGMENT:
        {
            CoreSegment* seg = static_cast<CoreSegment*>(segment);
            if(KyoukoRoot::useGpu)
            {
                KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "inputTransfers", error);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessInput", error, seg->numberOfNeuronSections, NEURONS_PER_NEURONSECTION);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessCoreSegment", error, 10, 64);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessOutput", error, seg->numberOfNeuronSections, 64);
                KyoukoRoot::gpuInterface->copyFromDevice(*(seg->data), "outputTransfers", error);
            }
            else if(KyoukoRoot::useCuda)
            {
                processing_CUDA(&seg->gpuPointer,
                                &seg->segmentSizes,
                                seg->brickOrder,
                                seg->brickHeaders,
                                seg->inputTransfers,
                                seg->outputTransfers,
                                seg->numberOfNeuronSections);
            }
            else
            {
                seg->segmentSettings->doLearn = 1;
                prcessCoreSegment(*seg);
                if(seg->segmentSettings->updateSections != 0) {
                    updateSections(*seg, false);
                }
                seg->segmentSettings->updateSections = 0;
            }

            seg->segmentSettings->doLearn = 0;
            break;
        }
        case INPUT_SEGMENT:
        {
            InputSegment* seg = static_cast<InputSegment*>(segment);
            prcessInputSegment(*seg);
            break;
        }
        case OUTPUT_SEGMENT:
        {
            OutputSegment* seg = static_cast<OutputSegment*>(segment);
            prcessOutputSegment(*seg);

            break;
        }
        default:
            break;
    }
}

/**
 * @brief run back-propagation on a segment
 *
 * @param segment segment to process
 */
void
CpuProcessingUnit::learnSegmentBackward(AbstractSegment* segment)
{
    Kitsunemimi::ErrorContainer error;

    switch(segment->getType())
    {
        case CORE_SEGMENT:
        {
            CoreSegment* seg = static_cast<CoreSegment*>(segment);

            if(KyoukoRoot::useGpu)
            {
                KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "inputTransfers", error);
                KyoukoRoot::gpuInterface->run(*(seg->data), "reweightOutput", error, seg->numberOfNeuronSections, NEURONS_PER_NEURONSECTION);
                KyoukoRoot::gpuInterface->run(*(seg->data), "reweightCoreSegment", error, seg->numberOfNeuronSections, 64);
                KyoukoRoot::gpuInterface->copyFromDevice(*(seg->data), "outputTransfers", error);
                KyoukoRoot::gpuInterface->copyFromDevice(*(seg->data), "updatePosSections", error);
                if(updateSections(*seg, true))
                {
                    KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "updatePosSections", error);
                    KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "neuronSections", error);
                    KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "synapseConnections", error);
                    KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "neuronConnections", error);
                }
                std::cout<<"counter: "<<counter<<std::endl;
                counter++;
            }
            else if(KyoukoRoot::useCuda)
            {
                backpropagation_CUDA(&seg->gpuPointer,
                                     &seg->segmentSizes,
                                     seg->brickOrder,
                                     seg->brickHeaders,
                                     seg->inputTransfers,
                                     seg->outputTransfers,
                                     seg->updatePosSections,
                                     seg->numberOfNeuronSections);

                if(updateSections(*seg, true))
                {
                    update_CUDA(&seg->gpuPointer,
                                &seg->segmentSizes,
                                seg->updatePosSections,
                                seg->neuronSections,
                                seg->synapseConnections,
                                seg->neuronConnections);
                }

                std::cout<<"counter: "<<counter<<std::endl;
                counter++;
            }
            else
            {
                reweightCoreSegment(*seg);
            }

            if(reductionCounter == 100) {
                //reduceNeurons(*seg);
                reductionCounter = 0;
            }
            reductionCounter++;
            break;
        }
        case OUTPUT_SEGMENT:
        {
            OutputSegment* seg = static_cast<OutputSegment*>(segment);
            backpropagateOutput(*seg);
            break;
        }
        default:
            break;
    }
}

/**
 * @brief process segments
 *
 * @param segment segment to process
 */
void
CpuProcessingUnit::processSegment(AbstractSegment* segment)
{
    Kitsunemimi::ErrorContainer error;

    switch(segment->getType())
    {
        case CORE_SEGMENT:
        {
            CoreSegment* seg = static_cast<CoreSegment*>(segment);
            if(KyoukoRoot::useGpu)
            {
                KyoukoRoot::gpuInterface->updateBufferOnDevice(*(seg->data), "inputTransfers", error);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessInput", error, seg->numberOfNeuronSections, NEURONS_PER_NEURONSECTION);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessCoreSegment", error, 10, 64);
                KyoukoRoot::gpuInterface->run(*(seg->data), "prcessOutput", error, seg->numberOfNeuronSections, 64);
                KyoukoRoot::gpuInterface->copyFromDevice(*(seg->data), "outputTransfers", error);
            }
            else if(KyoukoRoot::useCuda)
            {
                processing_CUDA(&seg->gpuPointer,
                                &seg->segmentSizes,
                                seg->brickOrder,
                                seg->brickHeaders,
                                seg->inputTransfers,
                                seg->outputTransfers,
                                seg->numberOfNeuronSections);
            }
            else
            {
                prcessCoreSegment(*seg);
            }

            break;
        }
        case INPUT_SEGMENT:
        {
            InputSegment* seg = static_cast<InputSegment*>(segment);
            prcessInputSegment(*seg);
            break;
        }
        case OUTPUT_SEGMENT:
        {
            OutputSegment* seg = static_cast<OutputSegment*>(segment);
            prcessOutputSegment(*seg);
            if(seg->parentCluster->msgClient == nullptr)
            {
                Task* actualTask = seg->parentCluster->getActualTask();
                const uint64_t cycle = actualTask->actualCycle;
                if(actualTask->type == IMAGE_REQUEST_TASK)
                {
                    // TODO: check for cluster-state instead of client
                    const uint32_t hightest = getHighestOutput(*seg);
                    Kitsunemimi::DataValue* value = actualTask->resultData->array[cycle]->toValue();
                    value->setValue(static_cast<long>(hightest));
                }
                else if(actualTask->type == TABLE_REQUEST_TASK)
                {
                    float val = 0.0f;
                    for(uint64_t i = 0; i < seg->segmentHeader->outputs.count; i++)
                    {
                        Kitsunemimi::DataValue* value = actualTask->resultData->array[cycle]->toValue();
                        val = value->getFloat() + seg->outputs[i].outputWeight;
                        value->setValue(val);
                    }
                }
            }
            break;
        }
        default:
            break;
    }
}

/**
 * @brief run loop to process all available segments
 */
void
CpuProcessingUnit::run()
{
    AbstractSegment* currentSegment = nullptr;

    while(m_abort == false)
    {
        currentSegment = KyoukoRoot::m_segmentQueue->getSegmentFromQueue();
        if(currentSegment != nullptr)
        {
            // check if segment is ready, else requeue
            if(currentSegment->isReady() == false)
            {
                KyoukoRoot::m_segmentQueue->addSegmentToQueue(currentSegment);
                continue;
            }

            // reset input ready status
            for(uint8_t side = 0; side < 16; side++) {
                currentSegment->segmentSlots->slots[side].inputReady = false;
            }

            // handle type of processing
            Cluster* clusterInterface = currentSegment->parentCluster;
            if(clusterInterface->mode == Cluster::LEARN_FORWARD_MODE) {
                learnSegmentForward(currentSegment);
            } else if(clusterInterface->mode == Cluster::LEARN_BACKWARD_MODE) {
                learnSegmentBackward(currentSegment);
            } else {
                processSegment(currentSegment);
            }

            // finish segment by sharing border-buffer and register in cluster
            currentSegment->finishSegment();
        }
        else
        {
            // if no segments are available then sleep
            sleepThread(1000);
        }
    }
}


/**
 * @brief SingleThreadProcessingStatic::reductionLearning

void
CpuProcessingUnit::reductionLearning(DynamicSegment* synapseSegment)
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

