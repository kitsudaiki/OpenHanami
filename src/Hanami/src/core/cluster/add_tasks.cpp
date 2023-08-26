/**
 * @file        add_tasks.cpp
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

#include "add_tasks.h"

#include <core/cluster/cluster.h>
#include <core/cluster/states/task_handle_state.h>
#include <core/cluster/statemachine_init.h>

#include <libKitsunemimiCommon/statemachine.h>

/**
 * @brief create a train-task and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputsPerCycle number of inputs per cycle
 * @param numberOfOuputsPerCycle number of outputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
addImageTrainTask(Cluster &cluster,
                  const std::string &name,
                  const std::string &userId,
                  const std::string &projectId,
                  float* inputData,
                  const uint64_t numberOfInputsPerCycle,
                  const uint64_t numberOfOuputsPerCycle,
                  const uint64_t numberOfCycles)
{
    // create new train-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.type = IMAGE_TRAIN_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputsPerCycle;
    newTask.numberOfOuputsPerCycle = numberOfOuputsPerCycle;

    // add task to queue
    const std::string uuid = newTask.uuid.toString();
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

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
addImageRequestTask(Cluster &cluster,
                    const std::string &name,
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
    for(uint64_t i = 0; i < numberOfCycles; i++) {
        newTask.resultData.append(0);
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
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to train table-data and add it to the task-queue
 *
 * @param inputData input-data
 * @param numberOfInputs number of inputs per cycle
 * @param numberOfCycles number of cycles
 *
 * @return task-uuid
 */
const std::string
addTableTrainTask(Cluster &cluster,
                  const std::string &name,
                  const std::string &userId,
                  const std::string &projectId,
                  float* inputData,
                  float* outputData,
                  const uint64_t numberOfInputs,
                  const uint64_t numberOfOutputs,
                  const uint64_t numberOfCycles)
{
    // create new train-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.inputData = inputData;
    newTask.outputData = outputData;
    newTask.type = TABLE_TRAIN_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.numberOfCycles = numberOfCycles;
    newTask.numberOfInputsPerCycle = numberOfInputs;
    newTask.numberOfOuputsPerCycle = numberOfOutputs;

    // add task to queue
    const std::string uuid = newTask.uuid.toString();
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

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
addTableRequestTask(Cluster &cluster,
                    const std::string &name,
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
    for(uint64_t i = 0; i < numberOfCycles; i++) {
        newTask.resultData.append(0.0f);
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
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

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
addClusterSnapshotSaveTask(Cluster &cluster,
                           const std::string &snapshotName,
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
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

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
addClusterSnapshotRestoreTask(Cluster &cluster,
                              const std::string &name,
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
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}
