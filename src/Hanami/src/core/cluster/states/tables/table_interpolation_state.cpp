/**
 * @file        table_interpolation_state.cpp
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

#include "table_interpolation_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TableInterpolation_State::TableInterpolation_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
TableInterpolation_State::~TableInterpolation_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
TableInterpolation_State::processEvent()
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
    for (uint64_t i = 0; i < numberOfInputsPerCycle; i++) {
        m_cluster->inputValues[i] = actualTask->inputData[(offset - numberOfInputsPerCycle) + i];
    }

    m_cluster->mode = ClusterProcessingMode::NORMAL_MODE;
    m_cluster->startForwardCycle();

    return true;
}
