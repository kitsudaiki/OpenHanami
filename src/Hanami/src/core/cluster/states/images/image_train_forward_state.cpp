/**
 * @file        image_train_forward_state.cpp
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

#include "image_train_forward_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
ImageTrainForward_State::ImageTrainForward_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
ImageTrainForward_State::~ImageTrainForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
ImageTrainForward_State::processEvent()
{
    Task* actualTask = m_cluster->getCurrentTask();
    const uint64_t numberOfInputsPerCycle = actualTask->numberOfInputsPerCycle;
    const uint64_t numberOfOuputsPerCycle = actualTask->numberOfOuputsPerCycle;
    const uint64_t entriesPerCycle = numberOfInputsPerCycle + numberOfOuputsPerCycle;
    const uint64_t offsetInput = entriesPerCycle * actualTask->actualCycle;

    // set input
    for (uint64_t i = 0; i < numberOfInputsPerCycle; i++) {
        m_cluster->inputValues[i] = actualTask->inputData[offsetInput + i];
    }

    // set exprected output
    for (uint64_t i = 0; i < numberOfOuputsPerCycle; i++) {
        const uint64_t numberOfCycles = numberOfInputsPerCycle;
        m_cluster->expectedValues[i] = actualTask->inputData[offsetInput + numberOfCycles + i];
    }

    m_cluster->mode = ClusterProcessingMode::TRAIN_FORWARD_MODE;
    m_cluster->startForwardCycle();

    return true;
}
