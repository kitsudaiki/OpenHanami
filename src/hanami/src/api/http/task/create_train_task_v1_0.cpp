/**
 * @file        create_train_task_v1_0.cpp
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

#include "create_train_task_v1_0.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/statemachine_init.h>
#include <database/cluster_table.h>
#include <database/dataset_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/statemachine.h>
#include <hanami_crypto/common.h>
#include <hanami_root.h>

CreateTrainTaskV1M0::CreateTrainTaskV1M0()
    : Blossom("Add new train-task to the task-queue of a cluster.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name", SAKURA_STRING_TYPE)
        .setComment("Name for the new task for better identification.")
        .setLimit(4, 254)
        .setRegex(NAME_REGEX);

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the cluster, which should process the request")
        .setRegex(UUID_REGEX);

    registerInputField("inputs", SAKURA_MAP_TYPE)
        .setComment(
            "key-value list with the names of the input-hexagons as key and the dataset-UUID, "
            "which "
            "should be used for the input, as value.");

    registerInputField("outputs", SAKURA_MAP_TYPE)
        .setComment(
            "key-value list with the names of the output-hexagons as key and the dataset-UUID, "
            "which "
            "should be used for the expected output, as value.");

    /*"inputs": {
        "test_hexagon": "asfd",
        "test_hexagon2": "asdf2"
    },
    "outputs": {
        "test_hexagon3": "asfd",
        "test_hexagon4": "asdf2"
    }*/

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the new created task.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the new created task.");

    registerOutputField("current_cycle", SAKURA_INT_TYPE)
        .setComment("Current cycle of the current task.");

    registerOutputField("total_number_of_cycles", SAKURA_INT_TYPE)
        .setComment("Total number of cycles requred by the task.");

    registerOutputField("state", SAKURA_STRING_TYPE)
        .setComment("Actual state of the task (queued, active, aborted or finished).");

    registerOutputField("queue_timestamp", SAKURA_STRING_TYPE)
        .setComment(
            "Timestamp in UTC when the task entered the queued state, "
            "which is basicall the timestamp when the task was created");

    registerOutputField("start_timestamp", SAKURA_STRING_TYPE)
        .setComment("Timestamp in UTC when the task entered the active state.");

    registerOutputField("end_timestamp", SAKURA_STRING_TYPE)
        .setComment("Timestamp in UTC when the task was finished.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateTrainTaskV1M0::runTask(BlossomIO& blossomIO,
                             const json& context,
                             BlossomStatus& status,
                             Hanami::ErrorContainer& error)
{
    const std::string taskName = blossomIO.input["name"];
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const Hanami::UserContext userContext = convertContext(context);
    const json inputs = blossomIO.input["inputs"];
    const json outputs = blossomIO.input["outputs"];

    // check if user exist within the table
    ClusterTable::ClusterDbEntry getResult;
    const ReturnStatus ret
        = ClusterTable::getInstance()->getCluster(getResult, clusterUuid, userContext, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Cluster with uuid '" + clusterUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get cluster
    Cluster* cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if (cluster == nullptr) {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "'not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // create new train-task
    Task* newTask = cluster->addNewTask();
    if (newTask == nullptr) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    newTask->name = taskName;
    newTask->userId = userContext.userId;
    newTask->projectId = userContext.projectId;
    newTask->type = TRAIN_TASK;
    newTask->progress.queuedTimeStamp = std::chrono::system_clock::now();
    newTask->info = TrainInfo();
    TrainInfo* info = &std::get<TrainInfo>(newTask->info);
    u_int64_t numberOfCycles = std::numeric_limits<uint64_t>::max();

    // prepare inputs
    for (const auto& [hexagonName, datasetUuid] : inputs.items()) {
        DataSetFileHandle fileHandle;
        if (fillTaskIo(fileHandle, userContext, hexagonName, datasetUuid, status, error) != OK) {
            return false;
        }
        if (numberOfCycles > fileHandle.header.numberOfRows) {
            numberOfCycles = fileHandle.header.numberOfRows;
        }
        info->inputs.try_emplace(hexagonName, std::move(fileHandle));
    }

    // prepare outputs
    for (const auto& [hexagonName, datasetUuid] : outputs.items()) {
        DataSetFileHandle fileHandle;
        if (fillTaskIo(fileHandle, userContext, hexagonName, datasetUuid, status, error) != OK) {
            return false;
        }
        if (numberOfCycles > fileHandle.header.numberOfRows) {
            numberOfCycles = fileHandle.header.numberOfRows;
        }
        info->outputs.try_emplace(hexagonName, std::move(fileHandle));
    }

    for (auto& [hexagonName, file_handle] : info->inputs) {
        file_handle.readSelector.endRow = numberOfCycles;
    }
    for (auto& [hexagonName, file_handle] : info->outputs) {
        file_handle.readSelector.endRow = numberOfCycles;
    }

    // set number of cycles
    newTask->progress.totalNumberOfCycles = numberOfCycles;
    info->numberOfCycles = numberOfCycles;

    cluster->stateMachine->goToNextState(PROCESS_TASK);

    // create output
    blossomIO.output["uuid"] = newTask->uuid.toString();
    blossomIO.output["name"] = taskName;
    blossomIO.output["current_cycle"] = newTask->progress.currentCyle;
    blossomIO.output["total_number_of_cycles"] = newTask->progress.totalNumberOfCycles;
    blossomIO.output["queue_timestamp"] = serializeTimePoint(newTask->progress.queuedTimeStamp);
    blossomIO.output["state"] = "queued";
    blossomIO.output["start_timestamp"] = "-";
    blossomIO.output["end_timestamp"] = "-";

    return true;
}

/**
 * @brief CreateTrainTask::fillTaskIo
 *
 * @param taskIo
 * @param userContext
 * @param settings
 * @param status reference to return status of the request
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
ReturnStatus
CreateTrainTaskV1M0::fillTaskIo(DataSetFileHandle& fileHandle,
                                const Hanami::UserContext& userContext,
                                const std::string& hexagonName,
                                const std::string& datasetUuid,
                                BlossomStatus& status,
                                Hanami::ErrorContainer& error)
{
    DataSetTable::DataSetDbEntry getResult;
    ReturnStatus ret
        = DataSetTable::getInstance()->getDataSet(getResult, datasetUuid, userContext, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return ret;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + datasetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return ret;
    }

    ret = openDataSetFile(fileHandle, getResult.location, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + datasetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
    }

    if (fileHandle.description.contains(hexagonName) == false) {
        status.errorMessage = "Dataset has no input for hexagon names '" + hexagonName + "'";
        status.statusCode = NOT_FOUND_RTYPE;
        return INVALID_INPUT;
    }

    fileHandle.readSelector.startColumn = fileHandle.description[hexagonName]["start_column"];
    fileHandle.readSelector.endColumn = fileHandle.description[hexagonName]["end_column"];

    return ret;
}
