/**
 * @file        task.h
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

#ifndef HANAMI_TASK_H
#define HANAMI_TASK_H

#include <common.h>

#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/items/data_items.h>

enum TaskType
{
    UNDEFINED_TASK = 0,
    IMAGE_TRAIN_TASK = 1,
    IMAGE_REQUEST_TASK = 2,
    TABLE_TRAIN_TASK = 3,
    TABLE_REQUEST_TASK = 4,
    CLUSTER_SNAPSHOT_SAVE_TASK = 5,
    CLUSTER_SNAPSHOT_RESTORE_TASK = 6,
};

enum TaskState
{
    UNDEFINED_TASK_STATE = 0,
    QUEUED_TASK_STATE = 1,
    ACTIVE_TASK_STATE = 2,
    ABORTED_TASK_STATE = 3,
    FINISHED_TASK_STATE = 4,
};

struct TaskProgress
{
    TaskState state = UNDEFINED_TASK_STATE;
    float percentageFinished = 0.0f;
    std::chrono::high_resolution_clock::time_point queuedTimeStamp;
    std::chrono::high_resolution_clock::time_point startActiveTimeStamp;
    std::chrono::high_resolution_clock::time_point endActiveTimeStamp;
    uint64_t estimatedRemaningTime = 0;
};

struct Task
{
    // task-identification
    kuuid uuid;
    TaskType type = UNDEFINED_TASK;
    std::string name = "";
    std::string userId = "";
    std::string projectId = "";

    // data-buffer
    float* inputData = nullptr;
    float* outputData = nullptr;
    Kitsunemimi::JsonItem resultData;

    // train-request-task meta
    uint64_t numberOfCycles = 0;
    uint64_t numberOfInputsPerCycle = 0;
    uint64_t numberOfOuputsPerCycle = 0;

    // snapshot-meta
    std::string snapshotName = "";
    std::string snapshotInfo = "";

    // progress
    uint64_t actualCycle = 0;
    TaskProgress progress;
};

#endif // HANAMI_TASK_H
