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

#include <hanami_root.h>
#include <database/request_result_table.h>
#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>

#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TaskHandle_State::TaskHandle_State(Cluster* cluster)
{
    m_cluster = cluster;
}

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
    Kitsunemimi::ErrorContainer error;
    m_task_mutex.lock();
    finishTask();
    const bool hasNextState = getNextTask();
    m_task_mutex.unlock();

    // handle empty queue
    if(hasNextState == false)
    {   
        // Azuki::setSpeedToMinimum(error);
        return true;
    }

    switch(actualTask->type)
    {
        case IMAGE_LEARN_TASK:
        {
            if(m_cluster->goToNextState(LEARN)) {
                m_cluster->goToNextState(IMAGE);
                // Azuki::setSpeedToAutomatic(error);
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case IMAGE_REQUEST_TASK:
        {
            if(m_cluster->goToNextState(REQUEST)) {
                m_cluster->goToNextState(IMAGE);
                // Azuki::setSpeedToAutomatic(error);
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case TABLE_LEARN_TASK:
        {
            if(m_cluster->goToNextState(LEARN)) {
                m_cluster->goToNextState(TABLE);
                // Azuki::setSpeedToAutomatic(error);
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case TABLE_REQUEST_TASK:
        {
            if(m_cluster->goToNextState(REQUEST)) {
                m_cluster->goToNextState(TABLE);
                // Azuki::setSpeedToAutomatic(error);
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case CLUSTER_SNAPSHOT_SAVE_TASK:
        {
            if(m_cluster->goToNextState(SNAPSHOT)) {
                if(m_cluster->goToNextState(CLUSTER)) {
                    m_cluster->goToNextState(SAVE);
                    // Azuki::setSpeedToAutomatic(error);
                } else {
                    // TODO: error-message
                    return false;
                }
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        case CLUSTER_SNAPSHOT_RESTORE_TASK:
        {
            if(m_cluster->goToNextState(SNAPSHOT)) {
                if(m_cluster->goToNextState(CLUSTER)) {
                    m_cluster->goToNextState(RESTORE);
                    // Azuki::setSpeedToAutomatic(error);
                } else {
                    // TODO: error-message
                    return false;
                }
            } else {
                // TODO: error-message
                return false;
            }
            break;
        }
        default: {
            // TODO: error-message
            // Azuki::setSpeedToMinimum(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief add new task
 *
 * @param uuid uuid of the new task for identification
 * @param task task itself
 *
 * @return alsways true
 */
bool
TaskHandle_State::addTask(const std::string &uuid,
                          const Task &task)
{
    std::lock_guard<std::mutex> guard(m_task_mutex);

    // TODO: check if uuid already exist (update comment)
    m_taskMap.insert(std::make_pair(uuid, task));
    m_taskQueue.push_back(uuid);

    return true;
}

/**
 * @brief get pointer to the actual active task
 *
 * @return pointer to the actual task of nullptr, if no task is active at the moment
 */
Task*
TaskHandle_State::getActualTask()
{
    std::lock_guard<std::mutex> guard(m_task_mutex);

    return actualTask;
}

/**
 * @brief run next task from the queue
 *
 * @return false, if task-queue if empty, else true
 */
bool
TaskHandle_State::getNextTask()
{
    // check number of tasks in queue
    if(m_taskQueue.size() == 0) {
        return false;
    }

    // remove task from queue
    const std::string nextUuid = m_taskQueue.front();
    m_taskQueue.pop_front();

    // init the new task
    std::map<std::string, Task>::iterator it;
    it = m_taskMap.find(nextUuid);
    it->second.progress.state = ACTIVE_TASK_STATE;
    it->second.progress.startActiveTimeStamp = std::chrono::system_clock::now();
    actualTask = &it->second;

    return true;
}

/**
 * @brief finish actual task
 */
void
TaskHandle_State::finishTask()
{
    // precheck
    if(actualTask == nullptr) {
        return;
    }

    // send results to shiori, if some are attached to the task
    if(actualTask->resultData != nullptr)
    {
        // results of tables a aggregated values, so they have to be fixed to its average value
        if(actualTask->type == TABLE_REQUEST_TASK)
        {
            const float numberOfOutputs = static_cast<float>(actualTask->numberOfOuputsPerCycle);
            float val = 0.0f;
            for(uint64_t i = 0; i < actualTask->resultData->array.size(); i++)
            {
                Kitsunemimi::DataValue* value = actualTask->resultData->array[i]->toValue();
                val = value->getFloat() / numberOfOutputs;
                value->setValue(val);
            }
        }

        // write result to database
        Kitsunemimi::ErrorContainer error;
        Kitsunemimi::JsonItem resultData;
        resultData.insert("uuid", actualTask->uuid.toString());
        resultData.insert("name", actualTask->name);
        resultData.insert("data", actualTask->resultData->copy());
        resultData.insert("visibility", "private");

        UserContext userContext;
        userContext.userId = actualTask->userId;
        userContext.projectId =actualTask->projectId;

        if(HanamiRoot::requestResultTable->addRequestResult(resultData, userContext, error) == false)
        {
            LOG_ERROR(error);
            return;
        }

        delete actualTask->resultData;
        actualTask->resultData = nullptr;
    }

    // remove task from map and free its data
    std::map<std::string, Task>::iterator it;
    it = m_taskMap.find(actualTask->uuid.toString());
    if(it != m_taskMap.end())
    {
        delete it->second.inputData;
        it->second.progress.state = FINISHED_TASK_STATE;
        it->second.progress.endActiveTimeStamp = std::chrono::system_clock::now();
    }

    actualTask = nullptr;
}

/**
 * @brief get task-progress
 *
 * @param taskUuid UUID of the task
 *
 * @return task-progress
 */
const TaskProgress
TaskHandle_State::getProgress(const std::string &taskUuid)
{
    std::lock_guard<std::mutex> guard(m_task_mutex);

    std::map<std::string, Task>::const_iterator it;
    it = m_taskMap.find(taskUuid);
    if(it != m_taskMap.end()) {
        return it->second.progress;
    }

    TaskProgress progress;
    return progress;
}

/**
 * @brief get state of a task
 *
 * @param taskUuid UUID of the task
 *
 * @return state of the requested task
 */
TaskState
TaskHandle_State::getTaskState(const std::string &taskUuid)
{
    TaskState state = UNDEFINED_TASK_STATE;

    std::map<std::string, Task>::const_iterator it;
    it = m_taskMap.find(taskUuid);
    if(it != m_taskMap.end()) {
        state = it->second.progress.state;
    }

    return state;
}

/**
 * @brief TaskHandle_State::getAllProgress
 * @param result
 */
void
TaskHandle_State::getAllProgress(std::map<std::string, TaskProgress> &result)
{
    for(auto const& [name, task] : m_taskMap) {
        result.emplace(task.uuid.toString(), task.progress);
    }
}

/**
 * @brief remove task from queue or abort the task, if actual in progress
 *
 * @param taskUuid UUID of the task
 *
 * @return false, if task-uuid was not found, else true
 */
bool
TaskHandle_State::removeTask(const std::string &taskUuid)
{
    std::lock_guard<std::mutex> guard(m_task_mutex);

    TaskState state = UNDEFINED_TASK_STATE;

    // check and update map
    std::map<std::string, Task>::iterator itMap;
    itMap = m_taskMap.find(taskUuid);
    if(itMap != m_taskMap.end())
    {
        state = itMap->second.progress.state;

        // if only queue but not activly processed at the moment, it can easily deleted
        if(state == QUEUED_TASK_STATE)
        {
            delete itMap->second.inputData;
            m_taskMap.erase(itMap);

            // update queue
            std::deque<std::string>::const_iterator itQueue;
            itQueue = std::find(m_taskQueue.begin(), m_taskQueue.end(), taskUuid);
            if(itQueue != m_taskQueue.end()) {
                m_taskQueue.erase(itQueue);
            }

            return true;
        }

        // if task is active at the moment, then only mark it as aborted
        if(state == ACTIVE_TASK_STATE)
        {
            itMap->second.progress.state = ABORTED_TASK_STATE;
            return true;
        }

        // handle finished and aborted state
        if(state == FINISHED_TASK_STATE
                || state == ABORTED_TASK_STATE)
        {
            // input-data are automatically deleted, when the task was finished,
            // so removing from the list is enough
            m_taskMap.erase(itMap);
            return true;
        }
    }

    return false;
}

/**
 * @brief check if a task is finished
 *
 * @param taskUuid UUID of the task
 *
 * @return true, if task is finished, else false
 */
bool
TaskHandle_State::isFinish(const std::string &taskUuid)
{
    std::lock_guard<std::mutex> guard(m_task_mutex);

    return getTaskState(taskUuid) == FINISHED_TASK_STATE;
}
