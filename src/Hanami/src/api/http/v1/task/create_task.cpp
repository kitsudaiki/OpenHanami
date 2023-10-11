/**
 * @file        create_image_train_task.cpp
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

#include "create_task.h"

#include <core/cluster/add_tasks.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <database/cluster_table.h>
#include <database/data_set_table.h>
#include <hanami_common/files/binary_file.h>
#include <hanami_crypto/common.h>
#include <hanami_files/data_set_files/data_set_functions.h>
#include <hanami_root.h>

CreateTask::CreateTask() : Blossom("Add new task to the task-queue of a cluster.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name", SAKURA_STRING_TYPE)
        .setComment("Name for the new task for better identification.")
        .setLimit(4, 256)
        .setRegex(NAME_REGEX);

    registerInputField("type", SAKURA_STRING_TYPE)
        .setComment("UUID of the data-set with the input, which coming from shiori.")
        .setRegex("^(train|request)$");

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the cluster, which should process the request")
        .setRegex(UUID_REGEX);

    registerInputField("data_set_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the data-set with the input, which coming from shiori.")
        .setRegex(UUID_REGEX);

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
CreateTask::runTask(BlossomIO &blossomIO,
                    const json &context,
                    BlossomStatus &status,
                    Hanami::ErrorContainer &error)
{
    const std::string name = blossomIO.input["name"];
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const std::string dataSetUuid = blossomIO.input["data_set_uuid"];
    const std::string taskType = blossomIO.input["type"];
    const UserContext userContext(context);

    // check if user exist within the table
    json getResult;
    if (ClusterTable::getInstance()->getCluster(getResult, clusterUuid, userContext, error)
        == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (getResult.size() == 0) {
        status.errorMessage = "Cluster with uuid '" + clusterUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get cluster
    Cluster *cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if (cluster == nullptr) {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "'not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get meta-infos of data-set from shiori
    json dataSetInfo;
    if (DataSetTable::getInstance()->getDateSetInfo(dataSetInfo, dataSetUuid, context, error)
        == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if (dataSetInfo.size() == 0) {
        status.errorMessage = "Data-set with uuid '" + dataSetUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // create task
    std::string taskUuid = "";
    if (dataSetInfo["type"] == "mnist") {
        imageTask(taskUuid, name, taskType, userContext, cluster, dataSetInfo, status, error);
    } else if (dataSetInfo["type"] == "csv") {
        tableTask(taskUuid, name, taskType, userContext, cluster, dataSetInfo, status, error);
    } else {
        status.errorMessage = "Invalid dataset-type '" + std::string(dataSetInfo["type"])
                              + "' given for to create new task";
        status.statusCode = BAD_REQUEST_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // create output
    blossomIO.output["uuid"] = taskUuid;
    blossomIO.output["name"] = name;

    return true;
}

/**
 * @brief add image-task to queue of cluster
 *
 * @param taskUuid reference for the output of the uuid of the new task
 * @param name name of the task
 * @param taskType type of the task (train or request)
 * @param userContext user-context
 * @param cluster pointer to the cluster, which should process the new task
 * @param dataSetInfo info-object with information about the dataset
 * @param status reference for status-output in error-case
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CreateTask::imageTask(std::string &taskUuid,
                      const std::string &name,
                      const std::string &taskType,
                      const UserContext &userContext,
                      Cluster *cluster,
                      json &dataSetInfo,
                      BlossomStatus &status,
                      Hanami::ErrorContainer &error)
{
    // get input-data
    const std::string dataSetLocation = dataSetInfo["location"];
    Hanami::DataBuffer buffer;
    if (getDataSetPayload(buffer, dataSetLocation, error) == false) {
        error.addMeesage("Failed to get data of data-set from location '" + dataSetLocation + "'");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get relevant information from output
    const uint64_t numberOfInputs = dataSetInfo["inputs"];
    const uint64_t numberOfOutputs = dataSetInfo["outputs"];
    const uint64_t numberOfLines = dataSetInfo["lines"];

    if (taskType == "train") {
        taskUuid = addImageTrainTask(*cluster,
                                     name,
                                     userContext.userId,
                                     userContext.projectId,
                                     static_cast<float *>(buffer.data),
                                     numberOfInputs,
                                     numberOfOutputs,
                                     numberOfLines);
    } else {
        taskUuid = addImageRequestTask(*cluster,
                                       name,
                                       userContext.userId,
                                       userContext.projectId,
                                       static_cast<float *>(buffer.data),
                                       numberOfInputs,
                                       numberOfOutputs,
                                       numberOfLines);
    }

    // detach buffer because the buffer-content is now attached to the task
    buffer.data = nullptr;

    return true;
}

/**
 * @brief add table-task to queue of cluster
 *
 * @param taskUuid reference for the output of the uuid of the new task
 * @param name name of the task
 * @param taskType type of the task (train or request)
 * @param userContext user-context
 * @param cluster pointer to the cluster, which should process the new task
 * @param dataSetInfo info-object with information about the dataset
 * @param status reference for status-output in error-case
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CreateTask::tableTask(std::string &taskUuid,
                      const std::string &name,
                      const std::string &taskType,
                      const UserContext &userContext,
                      Cluster *cluster,
                      json &dataSetInfo,
                      BlossomStatus &status,
                      Hanami::ErrorContainer &error)
{
    // init request-task
    const uint64_t numberOfInputs = cluster->clusterHeader->inputValues.count;
    const uint64_t numberOfOutputs = cluster->clusterHeader->outputValues.count;
    const uint64_t numberOfLines = dataSetInfo["lines"];

    // get input-data
    const std::string inputColumnName = "input";
    const std::string dataSetLocation = dataSetInfo["location"];
    Hanami::DataBuffer inputBuffer;
    if (getDataSetPayload(inputBuffer, dataSetLocation, error, inputColumnName) == false) {
        error.addMeesage("Failed to get data of data-set from location '" + dataSetLocation
                         + "' and column with name '" + inputColumnName + "'");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    if (taskType == "request") {
        taskUuid = addTableRequestTask(*cluster,
                                       name,
                                       userContext.userId,
                                       userContext.projectId,
                                       static_cast<float *>(inputBuffer.data),
                                       numberOfInputs,
                                       numberOfOutputs,
                                       numberOfLines - numberOfInputs);
        inputBuffer.data = nullptr;
    } else {
        // get output-data
        const std::string outputColumnName = "output";
        Hanami::DataBuffer outputBuffer;
        if (getDataSetPayload(outputBuffer, dataSetLocation, error, outputColumnName) == false) {
            error.addMeesage("Failed to get data of data-set from location '" + dataSetLocation
                             + "' and column with name '" + outputColumnName + "'");
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            return false;
        }

        // create task
        const uint64_t numberOfLines = dataSetInfo["lines"];
        taskUuid = addTableTrainTask(*cluster,
                                     name,
                                     userContext.userId,
                                     userContext.projectId,
                                     static_cast<float *>(inputBuffer.data),
                                     static_cast<float *>(outputBuffer.data),
                                     numberOfInputs,
                                     numberOfOutputs,
                                     numberOfLines - numberOfInputs);
        inputBuffer.data = nullptr;
        outputBuffer.data = nullptr;
    }

    return true;
}
