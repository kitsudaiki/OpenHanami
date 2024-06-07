/**
 * @file        task_handle_state.cpp
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

#include "task_handle_state.h"

#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>
#include <database/request_result_table.h>
#include <hanami_root.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TaskHandle_State::TaskHandle_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
TaskHandle_State::~TaskHandle_State() {}

/**
 * @brief prcess event
 *
 * @return false, if statechange failed, else true
 */
bool
TaskHandle_State::processEvent()
{
    Hanami::ErrorContainer error;
    const TaskType nextTaskType = m_cluster->finishTask();

    // handle empty queue
    if (nextTaskType == NO_TASK) {
        return true;
    }

    switch (nextTaskType) {
        case IMAGE_TRAIN_TASK:
        {
            if (m_cluster->goToNextState(TRAIN)) {
                m_cluster->goToNextState(IMAGE);
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case IMAGE_REQUEST_TASK:
        {
            if (m_cluster->goToNextState(REQUEST)) {
                m_cluster->goToNextState(IMAGE);
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case TABLE_TRAIN_TASK:
        {
            if (m_cluster->goToNextState(TRAIN)) {
                m_cluster->goToNextState(TABLE);
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case TABLE_REQUEST_TASK:
        {
            if (m_cluster->goToNextState(REQUEST)) {
                m_cluster->goToNextState(TABLE);
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case CLUSTER_CHECKPOINT_SAVE_TASK:
        {
            if (m_cluster->goToNextState(CHECKPOINT)) {
                if (m_cluster->goToNextState(CLUSTER)) {
                    m_cluster->goToNextState(SAVE);
                }
                else {
                    // TODO: error-message
                    return false;
                }
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case CLUSTER_CHECKPOINT_RESTORE_TASK:
        {
            if (m_cluster->goToNextState(CHECKPOINT)) {
                if (m_cluster->goToNextState(CLUSTER)) {
                    m_cluster->goToNextState(RESTORE);
                }
                else {
                    // TODO: error-message
                    return false;
                }
            }
            else {
                // TODO: error-message
                return false;
            }
            break;
        }
        default:
        {
            // TODO: error-message
            return false;
        }
    }

    return true;
}
