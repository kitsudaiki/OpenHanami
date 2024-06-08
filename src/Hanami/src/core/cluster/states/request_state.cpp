/**
 * @file        request_state.cpp
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

#include "request_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
Request_State::Request_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
Request_State::~Request_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
Request_State::processEvent()
{
    Task* actualTask = m_cluster->getCurrentTask();
    const RequestInfo info = std::get<RequestInfo>(actualTask->info);
    const uint64_t numberOfInputsPerCycle = info.numberOfInputsPerCycle;
    const uint64_t numberOfOuputsPerCycle = info.numberOfOuputsPerCycle;
    const uint64_t entriesPerCycle = numberOfInputsPerCycle + numberOfOuputsPerCycle;
    const uint64_t offsetInput = entriesPerCycle * actualTask->currentCycle;

    // set input
    InputInterface* inputInterface = &m_cluster->inputInterfaces.begin()->second;
    for (uint64_t i = 0; i < numberOfInputsPerCycle; i++) {
        inputInterface->inputNeurons[i].value = info.inputData[offsetInput + i];
    }

    m_cluster->mode = ClusterProcessingMode::NORMAL_MODE;
    m_cluster->startForwardCycle();

    return true;
}
