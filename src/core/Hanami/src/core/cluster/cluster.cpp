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
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/states/task_handle_state.h>
#include <core/processing/segment_queue.h>
#include <core/segments/output_segment/processing.h>
#include <api/websocket/cluster_io.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/statemachine.h>
#include <libKitsunemimiCommon/threading/thread.h>

/**
 * @brief constructor
 */
Cluster::Cluster()
{
    m_stateMachine = new Kitsunemimi::Statemachine();
    m_taskHandleState = new TaskHandle_State(this);

    initStatemachine(*m_stateMachine, this, m_taskHandleState);
}

/**
 * @brief destructor
 */
Cluster::~Cluster()
{
    delete m_stateMachine;

    // already deleted in the destructor of the statemachine
    // delete m_taskHandleState;

    for(AbstractSegment* segment : allSegments) {
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
Cluster::init(const Kitsunemimi::Hanami::ClusterMeta &clusterTemplate,
              const std::map<std::string, Kitsunemimi::Hanami::SegmentMeta> &segmentTemplates,
              const std::string &uuid)
{
    return initNewCluster(this, clusterTemplate, segmentTemplates, uuid);
}

/**
 * @brief Cluster::connectSlot
 *
 * @param sourceSegment
 * @param sourceSlotName
 * @param targetSegment
 * @param targetSlotName
 *
 * @return
 */
bool
Cluster::connectSlot(const std::string &sourceSegmentName,
                     const std::string &sourceSlotName,
                     const std::string &targetSegmentName,
                     const std::string &targetSlotName)
{
    const uint64_t sourceSegmentId = getSegmentId(sourceSegmentName);
    if(sourceSegmentId == UNINIT_STATE_64) {
        return false;
    }

    const uint64_t targetSegmentId = getSegmentId(targetSegmentName);
    if(targetSegmentId == UNINIT_STATE_64) {
        return false;
    }

    AbstractSegment* sourceSegment = allSegments.at(sourceSegmentId);
    AbstractSegment* targetSegment = allSegments.at(targetSegmentId);

    const uint8_t sourceSlotId = sourceSegment->getSlotId(sourceSlotName);
    if(sourceSlotId == UNINIT_STATE_8) {
        return false;
    }

    const uint8_t targetSlotId = targetSegment->getSlotId(targetSlotName);
    if(targetSlotId == UNINIT_STATE_8) {
        return false;
    }

    SegmentSlot* sourceSlot = &sourceSegment->segmentSlots->slots[sourceSlotId];
    SegmentSlot* targetSlot = &targetSegment->segmentSlots->slots[targetSlotId];

    sourceSlot->inUse = true;
    sourceSlot->targetSegmentId = targetSegmentId;
    sourceSlot->targetSlotId = targetSlotId;

    targetSlot->inUse = true;
    targetSlot->targetSegmentId = sourceSegmentId;
    targetSlot->targetSlotId = sourceSlotId;

    return true;
}

/**
 * @brief Cluster::getSegment
 * @param name
 * @return
 */
uint64_t
Cluster::getSegmentId(const std::string &name)
{
    for(uint64_t i = 0; i < allSegments.size(); i++)
    {
        if(allSegments.at(i)->getName() == name) {
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
    // set ready-states of all neighbors of all segments
    for(AbstractSegment* segment : allSegments)
    {
        for(uint8_t side = 0; side < 16; side++)
        {
            SegmentSlot* neighbor = &segment->segmentSlots->slots[side];
            // TODO: check possible crash here
            neighbor->inputReady = neighbor->direction != INPUT_DIRECTION;
        }
    }

    segmentCounter = 0;
    SegmentQueue::getInstance()->addSegmentListToQueue(allSegments);
}

/**
 * @brief start a new backward learn-cycle
 */
void
Cluster::startBackwardCycle()
{
    // set ready-states of all neighbors of all segments
    for(AbstractSegment* segment : allSegments)
    {
        for(uint8_t side = 0; side < 16; side++)
        {
            SegmentSlot* neighbor = &segment->segmentSlots->slots[side];
            neighbor->inputReady = neighbor->direction != OUTPUT_DIRECTION;
        }
    }

    segmentCounter = 0;
    SegmentQueue::getInstance()->addSegmentListToQueue(allSegments);
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
    if(segmentCounter < allSegments.size()) {
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
 * @brief create a learn-task and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputsPerCycle number of inputs per cycle
 * @param numberOfOuputsPerCycle number of outputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
Cluster::addImageLearnTask(const std::string &name,
                           const std::string &userId,
                           const std::string &projectId,
                           float* inputData,
                           const uint64_t numberOfInputsPerCycle,
                           const uint64_t numberOfOuputsPerCycle,
                           const uint64_t numberOfCycles)
{
    // create new learn-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.type = IMAGE_LEARN_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputsPerCycle;
    newTask.numberOfOuputsPerCycle = numberOfOuputsPerCycle;

    // add task to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create a request-task and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputsPerCycle number of inputs per cycle
 * @param numberOfOuputsPerCycle number of outputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
Cluster::addImageRequestTask(const std::string &name,
                             const std::string &userId,
                             const std::string &projectId,
                             float* inputData,
                             const uint64_t numberOfInputsPerCycle,
                             const uint64_t numberOfOuputsPerCycle,
                             const uint64_t numberOfCycles)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.resultData = new Kitsunemimi::DataArray();
    for(uint64_t i = 0; i < numberOfCycles; i++) {
        newTask.resultData->append(new Kitsunemimi::DataValue(0));
    }
    newTask.type = IMAGE_REQUEST_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputsPerCycle;
    newTask.numberOfOuputsPerCycle = numberOfOuputsPerCycle;

    // add task to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to learn table-data and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputs number of inputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
Cluster::addTableLearnTask(const std::string &name,
                           const std::string &userId,
                           const std::string &projectId,
                           float* inputData,
                           float* outputData,
                           const uint64_t numberOfInputs,
                           const uint64_t numberOfOutputs,
                           const uint64_t numberOfCycles)
{
    // create new learn-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.outputData = outputData;
    newTask.type = TABLE_LEARN_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputs;
    newTask.numberOfOuputsPerCycle = numberOfOutputs;

    // add task to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to request table-data and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputs number of inputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
Cluster::addTableRequestTask(const std::string &name,
                             const std::string &userId,
                             const std::string &projectId,
                             float* inputData,
                             const uint64_t numberOfInputs,
                             const uint64_t numberOfOutputs,
                             const uint64_t numberOfCycles)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.resultData = new Kitsunemimi::DataArray();
    for(uint64_t i = 0; i < numberOfCycles; i++) {
        newTask.resultData->append(new Kitsunemimi::DataValue(0.0f));
    }
    newTask.type = TABLE_REQUEST_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputs;
    newTask.numberOfOuputsPerCycle = numberOfOutputs;

    // add tasgetNextTaskk to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to create a snapshot from a cluster and add it to the task-queue
 *
 * @param snapshotName name for the snapshot
 * @param userId uuid of the user, where the snapshot belongs to
 * @param projectId uuid of the project, where the snapshot belongs to
 *
 * @return task-uuid
 */
const std::string
Cluster::addClusterSnapshotSaveTask(const std::string &snapshotName,
                                    const std::string &userId,
                                    const std::string &projectId)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = snapshotName;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.type = CLUSTER_SNAPSHOT_SAVE_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.snapshotName = snapshotName;

    // add tasgetNextTaskk to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to restore a cluster from a snapshot and add it to the task-queue
 *
 * @param snapshotUuid uuid of the snapshot
 * @param userId uuid of the user, where the snapshot belongs to
 * @param projectId uuid of the project, where the snapshot belongs to
 *
 * @return task-uuid
 */
const std::string
Cluster::addClusterSnapshotRestoreTask(const std::string &name,
                                       const std::string &snapshotInfo,
                                       const std::string &userId,
                                       const std::string &projectId)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.type = CLUSTER_SNAPSHOT_RESTORE_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.snapshotInfo = snapshotInfo;

    // add tasgetNextTaskk to queue
    const std::string uuid = newTask.uuid.toString();
    m_taskHandleState->addTask(uuid, newTask);

    m_stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief get actual task
 *
 * @return pointer to the actual task
 */
Task*
Cluster::getActualTask() const
{
    return m_taskHandleState->getActualTask();
}

/**
 * @brief get cycle of the actual task
 *
 * @return cycle of the actual task
 */
uint64_t
Cluster::getActualTaskCycle() const
{
    return m_taskHandleState->getActualTask()->actualCycle;
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
    return m_taskHandleState->getProgress(taskUuid);
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
    return m_taskHandleState->removeTask(taskUuid);
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
    return m_taskHandleState->isFinish(taskUuid);
}

/**
 * @brief Cluster::getAllProgress
 * @param result
 */
void
Cluster::getAllProgress(std::map<std::string, TaskProgress> &result)
{
    return m_taskHandleState->getAllProgress(result);
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
    return m_stateMachine->goToNextState(nextStateId);
}
