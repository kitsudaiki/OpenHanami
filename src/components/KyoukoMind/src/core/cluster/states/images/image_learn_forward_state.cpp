/**
 * @file        image_learn_forward_state.cpp
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

#include "image_learn_forward_state.h"

#include <core/segments/dynamic_segment/dynamic_segment.h>
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
ImageLearnForward_State::ImageLearnForward_State(Cluster* cluster)
{
    m_cluster = cluster;
}

/**
 * @brief destructor
 */
ImageLearnForward_State::~ImageLearnForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
ImageLearnForward_State::processEvent()
{
    Task* actualTask = m_cluster->getActualTask();
    const uint64_t numberOfInputsPerCycle = actualTask->getIntVal("number_of_inputs_per_cycle");
    const uint64_t numberOfOuputsPerCycle = actualTask->getIntVal("number_of_outputs_per_cycle");
    const uint64_t entriesPerCycle = numberOfInputsPerCycle + numberOfOuputsPerCycle;
    const uint64_t offsetInput = entriesPerCycle * actualTask->actualCycle;

    // set input
    InputNeuron* inputNeurons = m_cluster->inputSegments.begin()->second->inputs;
    for(uint64_t i = 0; i < numberOfInputsPerCycle; i++) {
        inputNeurons[i].weight = actualTask->inputData[offsetInput + i];
    }

    // set exprected output
    OutputNeuron* outputNeurons = m_cluster->outputSegments.begin()->second->outputs;
    for(uint64_t i = 0; i < numberOfOuputsPerCycle; i++)
    {
        const uint64_t numberOfCycles = numberOfInputsPerCycle;
        outputNeurons[i].shouldValue = actualTask->inputData[offsetInput + numberOfCycles + i];
    }

    m_cluster->mode = Cluster::LEARN_FORWARD_MODE;
    m_cluster->startForwardCycle();

    return true;
}
