/**
 * @file        table_learn_forward_state.cpp
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

#include "table_learn_forward_state.h"

#include <core/segments/core_segment/core_segment.h>
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TableLearnForward_State::TableLearnForward_State(Cluster* cluster)
{
    m_cluster = cluster;
}

/**
 * @brief destructor
 */
TableLearnForward_State::~TableLearnForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
TableLearnForward_State::processEvent()
{
    Task* actualTask = m_cluster->getActualTask();
    const uint64_t numberOfInputsPerCycle = actualTask->getIntVal("number_of_inputs_per_cycle");
    const uint64_t numberOfOuputsPerCycle = actualTask->getIntVal("number_of_outputs_per_cycle");
    uint64_t offset = actualTask->actualCycle;
    if(numberOfInputsPerCycle > numberOfOuputsPerCycle) {
        offset += numberOfInputsPerCycle;
    } else {
        offset += numberOfOuputsPerCycle;
    }

    // set input
    InputNeuron* inputNeurons = m_cluster->inputSegments.begin()->second->inputs;
    for(uint64_t i = 0; i < numberOfInputsPerCycle; i++) {
        inputNeurons[i].weight = actualTask->inputData[(offset - numberOfInputsPerCycle) + i];
    }

    // set exprected output
    OutputNeuron* outputNeurons = m_cluster->outputSegments.begin()->second->outputs;
    for(uint64_t i = 0; i < numberOfOuputsPerCycle; i++) {
        outputNeurons[i].shouldValue = actualTask->outputData[(offset - numberOfOuputsPerCycle) + i];
    }

    m_cluster->mode = Cluster::LEARN_FORWARD_MODE;
    m_cluster->startForwardCycle();

    return true;
}
