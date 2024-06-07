/**
 * @file        cycle_finish_state.cpp
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

#include "cycle_finish_state.h"

#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/task.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
CycleFinish_State::CycleFinish_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
CycleFinish_State::~CycleFinish_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
CycleFinish_State::processEvent()
{
    uint64_t numberOfCycles = 0;
    Task* actualTask = m_cluster->getCurrentTask();
    switch (actualTask->type) {
        case TRAIN_TASK:
            numberOfCycles = std::get<TrainInfo>(actualTask->info).numberOfCycles;
            break;
        case REQUEST_TASK:
            numberOfCycles = std::get<RequestInfo>(actualTask->info).numberOfCycles;
            break;
        case CLUSTER_CHECKPOINT_SAVE_TASK:
            return false;
        case CLUSTER_CHECKPOINT_RESTORE_TASK:
            return false;
        case NO_TASK:
            return false;
    }

    // update progress-counter
    actualTask->currentCycle++;
    actualTask->progress.currentCyle = actualTask->currentCycle;

    // to go next state of finish the task to goal is reached
    if (actualTask->currentCycle == numberOfCycles) {
        m_cluster->goToNextState(FINISH_TASK);
    }
    else {
        m_cluster->goToNextState(NEXT);
    }

    return true;
}
