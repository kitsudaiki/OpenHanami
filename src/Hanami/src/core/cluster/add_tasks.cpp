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

#include <hanami_common/statemachine.h>

/**
 * @brief create image training task
 *
 * @param cluster reference to the cluster, which should run the task
 * @param name name of the task
 * @param userId id of the user, who started the task
 * @param projectId id of the project, where the user is
 * @param inputData pointer to the input training-data
 * @param numberOfInputsPerCycle number of input-values per iteration
 * @param numberOfOuputsPerCycle number of output-values per iteration
 * @param numberOfCycles number of iterations
 *
 * @return uuid of the new task
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
 * @brief create image request task
 *
 * @param cluster reference to the cluster, which should run the task
 * @param name name of the task
 * @param userId id of the user, who started the task
 * @param projectId id of the project, where the user is
 * @param inputData pointer to the input test-data
 * @param numberOfInputsPerCycle number of input-values per iteration
 * @param numberOfOuputsPerCycle number of output-values per iteration
 * @param numberOfCycles number of iterations
 *
 * @return uuid of the new task
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
        newTask.resultData.push_back(0);
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
 * @brief create table training task
 *
 * @param cluster reference to the cluster, which should run the task
 * @param name name of the task
 * @param userId id of the user, who started the task
 * @param projectId id of the project, where the user is
 * @param inputData pointer to the input training-data
 * @param outputData pointer to the output training-data
 * @param numberOfInputsPerCycle number of input-values per iteration
 * @param numberOfOuputsPerCycle number of output-values per iteration
 * @param numberOfCycles number of iterations
 *
 * @return uuid of the new task
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
 * @brief create table request task
 *
 * @param cluster reference to the cluster, which should run the task
 * @param name name of the task
 * @param userId id of the user, who started the task
 * @param projectId id of the project, where the user is
 * @param inputData pointer to the input test-data
 * @param numberOfInputsPerCycle number of input-values per iteration
 * @param numberOfOuputsPerCycle number of output-values per iteration
 * @param numberOfCycles number of iterations
 *
 * @return uuid of the new task
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
        newTask.resultData.push_back(0.0f);
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
 * @brief create task to create a checkpoint from a cluster and add it to the task-queue
 *
 * @param cluster reference to the cluster, which should run the task
 * @param checkpointName name for the checkpoint
 * @param userId uuid of the user, where the checkpoint belongs to
 * @param projectId uuid of the project, where the checkpoint belongs to
 *
 * @return uuid of the new task
 */
const std::string
addCheckpointSaveTask(Cluster &cluster,
                      const std::string &checkpointName,
                      const std::string &userId,
                      const std::string &projectId)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = checkpointName;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.type = CLUSTER_CHECKPOINT_SAVE_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.checkpointName = checkpointName;

    // add tasgetNextTaskk to queue
    const std::string uuid = newTask.uuid.toString();
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}

/**
 * @brief create task to restore a cluster from a checkpoint and add it to the task-queue
 *
 * @param cluster reference to the cluster, which should run the task
 * @param name name of the new checkpoint
 * @param checkpointUuid uuid of the checkpoint
 * @param userId uuid of the user, where the checkpoint belongs to
 * @param projectId uuid of the project, where the checkpoint belongs to
 *
 * @return uuid of the new task
 */
const std::string
addCheckpointRestoreTask(Cluster &cluster,
                         const std::string &name,
                         const std::string &checkpointInfo,
                         const std::string &userId,
                         const std::string &projectId)
{
    // create new request-task
    Task newTask;
    newTask.uuid = generateUuid();
    newTask.name = name;
    newTask.userId = userId;
    newTask.projectId = projectId;
    newTask.type = CLUSTER_CHECKPOINT_RESTORE_TASK;
    newTask.progress.state = QUEUED_TASK_STATE;
    newTask.progress.queuedTimeStamp = std::chrono::system_clock::now();

    // fill metadata
    newTask.checkpointInfo = checkpointInfo;

    // add tasgetNextTaskk to queue
    const std::string uuid = newTask.uuid.toString();
    cluster.taskHandleState->addTask(uuid, newTask);

    cluster.stateMachine->goToNextState(PROCESS_TASK);

    return uuid;
}
