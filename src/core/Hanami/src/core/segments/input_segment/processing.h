/**
 * @file        processing.h
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

#ifndef HANAMI_INPUT_PROCESSING_H
#define HANAMI_INPUT_PROCESSING_H

#include <common.h>

#include <hanami_root.h>
#include "objects.h"
#include "input_segment.h"

/**
 * @brief process all neurons within a specific brick and also all synapse-sections,
 *        which are connected to an active neuron
 *
 * @param segment input-segment to process
 */
void
prcessInputSegment(const InputSegment &segment)
{
    InputNeuron* neuron = nullptr;
    const uint64_t numberOfInputs = segment.segmentHeader->inputs.count;
    float* outputTransfers = segment.outputTransfers;

    for(uint64_t pos = 0; pos < numberOfInputs; pos++)
    {
        neuron = &segment.inputs[pos];
        outputTransfers[neuron->targetBorderId] = neuron->weight;
    }
}

#endif // HANAMI_INPUT_PROCESSING_H
