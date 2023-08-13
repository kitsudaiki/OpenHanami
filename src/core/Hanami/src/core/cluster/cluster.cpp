/**
 * @file        cluster.cpp
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

#include "cluster.h"
#include <hanami_root.h>

#include <core/segments/core_segment/core_segment.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/states/task_handle_state.h>
#include <core/processing/segment_queue.h>
#include <api/websocket/cluster_io.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/statemachine.h>
#include <libKitsunemimiCommon/threading/thread.h>

/**
 * @brief constructor
 */
Cluster::Cluster()
{
    stateMachine = new Kitsunemimi::Statemachine();
    taskHandleState = new TaskHandle_State(this);

    initStatemachine(*stateMachine, this, taskHandleState);
}

/**
 * @brief destructor
 */
Cluster::~Cluster()
{
    delete stateMachine;

    // already deleted in the destructor of the statemachine
    // delete m_taskHandleState;

    for(CoreSegment* segment : coreSegments) {
        delete segment;
    }
}

/**
 * @brief get uuid of the cluster
 *
 * @return uuid of the cluster
 */
const
std::string Cluster::getUuid()
{
    return networkMetaData->uuid.toString();
}

/**
 * @brief init the cluster
 *
 * @param parsedContent TODO
 * @param segmentTemplates TODO
 * @param uuid UUID of the new cluster
 *
 * @return true, if successful, else false
 */
bool
Cluster::init(const Kitsunemimi::Hanami::SegmentMeta &clusterTemplate,
              const std::string &uuid)
{
    return initNewCluster(this, clusterTemplate, uuid);
}

/**
 * @brief Cluster::getSegment
 * @param name
 * @return
 */
uint64_t
Cluster::getSegmentId(const std::string &name)
{
    for(uint64_t i = 0; i < coreSegments.size(); i++)
    {
        if(coreSegments.at(i)->getName() == name) {
            return i;
        }
    }

    return UNINIT_STATE_64;
}

/**
 * @brief get the name of the clsuter
 *
 * @return name of the cluster
 */
const std::string
Cluster::getName()
{
    // precheck
    if(networkMetaData == nullptr) {
        return std::string("");
    }

    return std::string(networkMetaData->name);
}

/**
 * @brief set new name for the cluster
 *
 * @param newName new name
 *
 * @return true, if successful, else false
 */
bool
Cluster::setName(const std::string newName)
{
    // precheck
    if(networkMetaData == nullptr
            || newName.size() > 1023
            || newName.size() == 0)
    {
        return false;
    }

    // copy string into char-buffer and set explicit the escape symbol to be absolut sure
    // that it is set to absolut avoid buffer-overflows
    strncpy(networkMetaData->name, newName.c_str(), newName.size());
    networkMetaData->name[newName.size()] = '\0';

    return true;
}

/**
 * @brief start a new forward learn-cycle
 */
void
Cluster::startForwardCycle()
{
    segmentCounter = 0;
    SegmentQueue::getInstance()->addSegmentListToQueue(coreSegments);
}

/**
 * @brief start a new backward learn-cycle
 */
void
Cluster::startBackwardCycle()
{
    segmentCounter = 0;
    SegmentQueue::getInstance()->addSegmentListToQueue(coreSegments);
}

/**
 * @brief switch state of the cluster between task and direct mode
 *
 * @param newState new desired state
 *
 * @return true, if switch in statemachine was successful, else false
 */
bool
Cluster::setClusterState(const std::string &newState)
{
    if(newState == "DIRECT") {
        return goToNextState(SWITCH_TO_DIRECT_MODE);
    }

    if(newState == "TASK") {
        return goToNextState(SWITCH_TO_TASK_MODE);
    }

    return false;
}

/**
 * @brief update state of the cluster, which is caled for each finalized segment
 */
void
Cluster::updateClusterState()
{
    std::lock_guard<std::mutex> guard(m_segmentCounterLock);

    segmentCounter++;
    if(segmentCounter < coreSegments.size()) {
        return;
    }

    // trigger next lerning phase, if already in phase 1
    if(mode == Cluster::LEARN_FORWARD_MODE)
    {
        mode = Cluster::LEARN_BACKWARD_MODE;
        startBackwardCycle();
        return;
    }

    // send message, that process was finished
    if(mode == Cluster::LEARN_BACKWARD_MODE) {
        sendClusterLearnEndMessage(this);
    } else if(mode == Cluster::NORMAL_MODE) {
        sendClusterNormalEndMessage(this);
    }

    goToNextState(NEXT);
}


/**
 * @brief get actual task
 *
 * @return pointer to the actual task
 */
Task*
Cluster::getActualTask() const
{
    return taskHandleState->getActualTask();
}

/**
 * @brief get cycle of the actual task
 *
 * @return cycle of the actual task
 */
uint64_t
Cluster::getActualTaskCycle() const
{
    return taskHandleState->getActualTask()->actualCycle;
}

/**
 * @brief get task-progress
 *
 * @param taskUuid UUID of the task
 *
 * @return task-progress
 */
const TaskProgress
Cluster::getProgress(const std::string &taskUuid)
{
    return taskHandleState->getProgress(taskUuid);
}

/**
 * @brief remove task from queue of abort the task, if actual in progress
 *
 * @param taskUuid UUID of the task
 *
 * @return always true
 */
bool
Cluster::removeTask(const std::string &taskUuid)
{
    return taskHandleState->removeTask(taskUuid);
}

/**
 * @brief check if a task is finished
 *
 * @param taskUuid UUID of the task
 *
 * @return true, if task is finished, else false
 */
bool
Cluster::isFinish(const std::string &taskUuid)
{
    return taskHandleState->isFinish(taskUuid);
}

/**
 * @brief Cluster::getAllProgress
 * @param result
 */
void
Cluster::getAllProgress(std::map<std::string, TaskProgress> &result)
{
    return taskHandleState->getAllProgress(result);
}

/**
 * @brief switch statemachine of cluster to next state
 *
 * @param nextStateId id of the next state
 *
 * @return true, if statemachine switch was successful, else false
 */
bool
Cluster::goToNextState(const uint32_t nextStateId)
{
    return stateMachine->goToNextState(nextStateId);
}
