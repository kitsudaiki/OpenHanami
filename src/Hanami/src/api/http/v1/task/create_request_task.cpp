/**
 * @file        create_request_task.cpp
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

#include "create_request_task.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/statemachine_init.h>
#include <database/cluster_table.h>
#include <database/dataset_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_common/statemachine.h>
#include <hanami_config/config_handler.h>
#include <hanami_crypto/common.h>
#include <hanami_root.h>

CreateRequestTask::CreateRequestTask()
    : Blossom("Add new request-task to the task-queue of a cluster.")
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

    registerInputField("number_of_cycles", SAKURA_INT_TYPE)
        .setComment("Number of cycles")
        .setLimit(0, 1000000000);

    registerInputField("inputs", SAKURA_MAP_TYPE)
        .setComment("UUID of the dataset with the input, which coming from shiori.");

    registerInputField("results", SAKURA_MAP_TYPE)
        .setComment("UUID of the dataset with the input, which coming from shiori.");

    /*inputs: {
        test_brick: {
            dataset_uuid: asfd,
            start_row: 0,
            start_column: 0,
            end_column: 50,
        },
        test_brick2: {
            dataset_uuid: poi,
            start_row: 0,
            start_column: 0,
            end_column: 100,
        }
    },
    results: {
        test_brick_out: {
            name: poi
        }
        test_brick_out2: {
            name: poi2
        }
    }*/

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the new created task.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the new created task.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateRequestTask::runTask(BlossomIO& blossomIO,
                           const json& context,
                           BlossomStatus& status,
                           Hanami::ErrorContainer& error)
{
    const std::string name = blossomIO.input["name"];
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const Hanami::UserContext userContext = convertContext(context);
    const uint64_t numberOfCycles = blossomIO.input["number_of_cycles"];

    // check if user exist within the table
    json getResult;
    ReturnStatus ret = ClusterTable::getInstance()->getCluster(
        getResult, clusterUuid, userContext, false, error);
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

    // create new request-task
    Task* newTask = cluster->addNewTask();
    if (newTask == nullptr) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    newTask->name = name;
    newTask->userId = userContext.userId;
    newTask->projectId = userContext.projectId;
    newTask->type = REQUEST_TASK;
    newTask->progress.queuedTimeStamp = std::chrono::system_clock::now();
    newTask->progress.totalNumberOfCycles = numberOfCycles;
    newTask->info = RequestInfo();

    RequestInfo* info = &std::get<RequestInfo>(newTask->info);
    info->numberOfCycles = numberOfCycles;

    // prepare input
    const json inputs = blossomIO.input["inputs"];
    for (const auto& [brickName, settings] : inputs.items()) {
        DataSetFileHandle fileHandle;

        if (settings.contains("dataset_uuid") == false) {
            status.statusCode = BAD_REQUEST_RTYPE;
            return INVALID_INPUT;
        }

        if (fillTaskIo(fileHandle, userContext, settings, numberOfCycles, status, error) != OK) {
            return false;
        }
        info->inputs.try_emplace(brickName, std::move(fileHandle));
    }

    // prepare result
    const json results = blossomIO.input["results"];
    for (const auto& [brickName, settings] : results.items()) {
        if (settings.contains("name") == false) {
            status.statusCode = BAD_REQUEST_RTYPE;
            return INVALID_INPUT;
        }

        const uint64_t numberOfOutputs = cluster->outputInterfaces[brickName].outputNeurons.size();
        const std::string name = settings["name"];

        DataSetFileHandle fileHandle;
        const std::string datasetUuid = newTask->uuid.toString();
        if (createResultTarget(
                fileHandle, datasetUuid, name, userContext, numberOfOutputs, status, error)
            != OK)
        {
            return false;
        }
        info->results.try_emplace(brickName, std::move(fileHandle));
    }

    cluster->stateMachine->goToNextState(PROCESS_TASK);

    // create output
    blossomIO.output["uuid"] = newTask->uuid.toString();
    blossomIO.output["name"] = name;

    return true;
}

/**
 * @brief CreateTrainTask::fillTaskIo
 *
 * @param taskIo
 * @param userContext
 * @param settings
 * @param numberOfCycles number of cycles of the request
 * @param status reference to return status of the request
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
ReturnStatus
CreateRequestTask::fillTaskIo(DataSetFileHandle& fileHandle,
                              const Hanami::UserContext& userContext,
                              const json& settings,
                              const uint64_t numberOfCycles,
                              BlossomStatus& status,
                              Hanami::ErrorContainer& error)
{
    ReturnStatus ret = fileHandle.readSelector.fromJson(settings);
    if (ret != OK) {
        status.statusCode = BAD_REQUEST_RTYPE;
        return INVALID_INPUT;
    }
    fileHandle.readSelector.endRow = fileHandle.readSelector.startRow + numberOfCycles;

    if (settings.contains("dataset_uuid") == false) {
        status.statusCode = BAD_REQUEST_RTYPE;
        return INVALID_INPUT;
    }

    const std::string datasetUuid = settings["dataset_uuid"];
    DataSetTable::DataSetDbEntry getResult;
    ret = DataSetTable::getInstance()->getDataSet(getResult, datasetUuid, userContext, error);
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

    return ret;
}

/**
 * @brief CreateRequestTask::createResultTarget
 * @param datasetUuid
 * @param name
 * @param userContext
 * @param status
 * @param error
 * @return
 */
ReturnStatus
CreateRequestTask::createResultTarget(DataSetFileHandle& fileHandle,
                                      const std::string& datasetUuid,
                                      const std::string& name,
                                      const Hanami::UserContext& userContext,
                                      const uint64_t numberOfOutputs,
                                      BlossomStatus& status,
                                      Hanami::ErrorContainer& error)
{
    bool success = false;
    std::string targetFilePath = GET_STRING_CONFIG("storage", "dataset_location", success);
    if (success == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMessage("file-location to store dataset is missing in the config");
        return ERROR;
    }

    if (targetFilePath.at(targetFilePath.size() - 1) != '/') {
        targetFilePath.append("/");
    }
    targetFilePath.append(name + "_result_" + datasetUuid);

    DataSetTable::DataSetDbEntry dbEntry;
    dbEntry.name = name + "_result";
    dbEntry.ownerId = userContext.userId;
    dbEntry.projectId = userContext.projectId;
    dbEntry.uuid = datasetUuid;
    dbEntry.visibility = "private";
    dbEntry.type = "csv";
    dbEntry.location = targetFilePath;

    if (DataSetTable::getInstance()->addDataSet(dbEntry, userContext, error) != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return ERROR;
    }

    ReturnStatus ret
        = initNewDataSetFile(fileHandle, targetFilePath, name, FLOAT_TYPE, numberOfOutputs, error);
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Data-set with uuid '" + datasetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
    }

    return OK;
}
