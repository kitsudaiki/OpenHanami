/**
 * @file        table_train_forward_state.cpp
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

#include "table_train_forward_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TableTrainForward_State::TableTrainForward_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
TableTrainForward_State::~TableTrainForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
TableTrainForward_State::processEvent()
{
    Task* actualTask = m_cluster->getCurrentTask();
    const uint64_t numberOfInputsPerCycle = actualTask->numberOfInputsPerCycle;
    const uint64_t numberOfOuputsPerCycle = actualTask->numberOfOuputsPerCycle;
    uint64_t offset = actualTask->actualCycle;
    if (numberOfInputsPerCycle > numberOfOuputsPerCycle) {
        offset += numberOfInputsPerCycle;
    }
    else {
        offset += numberOfOuputsPerCycle;
    }

    // set input
    InputInterface* inputInterface = &m_cluster->inputInterfaces.begin()->second;
    for (uint64_t i = 0; i < actualTask->numberOfInputsPerCycle; i++) {
        inputInterface->inputNeurons[i].value
            = actualTask->inputData[(offset - numberOfInputsPerCycle) + i];
    }

    // set exprected output
    OutputInterface* outputInterface = &m_cluster->outputInterfaces.begin()->second;
    for (uint64_t i = 0; i < actualTask->numberOfInputsPerCycle; i++) {
        outputInterface->outputNeurons[i].exprectedVal
            = actualTask->outputData[(offset - numberOfOuputsPerCycle) + i];
    }

    m_cluster->mode = ClusterProcessingMode::TRAIN_FORWARD_MODE;
    m_cluster->startForwardCycle();

    return true;
}
