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

#include <database/checkpoint_table.h>
#include <hanami_common/uuid.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <variant>

using json = nlohmann::json;

enum TaskType {
    NO_TASK = 0,
    TRAIN_TASK = 1,
    REQUEST_TASK = 2,
    CLUSTER_CHECKPOINT_SAVE_TASK = 3,
    CLUSTER_CHECKPOINT_RESTORE_TASK = 4,
};

enum TaskState {
    UNDEFINED_TASK_STATE = 0,
    QUEUED_TASK_STATE = 1,
    ACTIVE_TASK_STATE = 2,
    ABORTED_TASK_STATE = 3,
    FINISHED_TASK_STATE = 4,
};

struct TaskProgress {
    TaskState state = UNDEFINED_TASK_STATE;
    uint64_t totalNumberOfCycles = 0;
    uint64_t currentCyle = 0;
    std::chrono::high_resolution_clock::time_point queuedTimeStamp;
    std::chrono::high_resolution_clock::time_point startActiveTimeStamp;
    std::chrono::high_resolution_clock::time_point endActiveTimeStamp;
    uint64_t estimatedRemaningTime = 0;
};

struct TrainInfo {
    float* inputData = nullptr;
    float* outputData = nullptr;

    uint64_t numberOfCycles = 0;
    uint64_t numberOfInputsPerCycle = 0;
    uint64_t numberOfOuputsPerCycle = 0;
};

struct RequestInfo {
    float* inputData = nullptr;

    uint64_t numberOfCycles = 0;
    uint64_t numberOfInputsPerCycle = 0;
    uint64_t numberOfOuputsPerCycle = 0;
};

struct CheckpointSaveInfo {
    std::string checkpointName = "";
};

struct CheckpointRestoreInfo {
    CheckpointTable::CheckpointDbEntry checkpointInfo;
};

struct Task {
    UUID uuid;
    TaskType type = NO_TASK;
    std::string name = "";
    std::string userId = "";
    std::string projectId = "";

    // progress
    uint64_t currentCycle = 0;
    TaskProgress progress;
    json resultData;

    std::variant<TrainInfo, RequestInfo, CheckpointSaveInfo, CheckpointRestoreInfo> info;

    Task()
    {
        uuid = generateUuid();
        progress.state = QUEUED_TASK_STATE;
        progress.queuedTimeStamp = std::chrono::system_clock::now();
    }

    void deleteData()
    {
        switch (type) {
            case TRAIN_TASK:
                if (std::get<TrainInfo>(info).inputData != nullptr) {
                    delete[] std::get<TrainInfo>(info).inputData;
                    std::get<TrainInfo>(info).inputData = nullptr;
                }
                if (std::get<TrainInfo>(info).outputData != nullptr) {
                    delete[] std::get<TrainInfo>(info).outputData;
                    std::get<TrainInfo>(info).outputData = nullptr;
                }
                break;
            case REQUEST_TASK:
                if (std::get<RequestInfo>(info).inputData != nullptr) {
                    delete[] std::get<RequestInfo>(info).inputData;
                    std::get<RequestInfo>(info).inputData = nullptr;
                }
                break;
            case CLUSTER_CHECKPOINT_SAVE_TASK:
                return;
            case CLUSTER_CHECKPOINT_RESTORE_TASK:
                return;
            case NO_TASK:
                return;
        }
    }

    ~Task()
    {
        // deleteData();
    }
};

#endif  // HANAMI_TASK_H
