/**
 * @file        create_image_learn_task.cpp
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
#include <hanami_root.h>
#include <database/data_set_table.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>
#include <core/data_set_files/data_set_functions.h>

#include <libKitsunemimiHanamiCommon/enums.h>

#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiCrypto/common.h>

using namespace Kitsunemimi::Hanami;

CreateTask::CreateTask()
    : Blossom("Add new task to the task-queue of a cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name for the new task for better identification.");
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", NAME_REGEX));

    registerInputField("type",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the data-set with the input, which coming from shiori.");
    assert(addFieldRegex("type", "^(learn|request)$"));

    registerInputField("cluster_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster, which should process the request");
    assert(addFieldRegex("cluster_uuid", UUID_REGEX));

    registerInputField("data_set_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the data-set with the input, which coming from shiori.");
    assert(addFieldRegex("data_set_uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new created task.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new created task.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateTask::runTask(BlossomIO &blossomIO,
                              const Kitsunemimi::DataMap &context,
                              BlossomStatus &status,
                              Kitsunemimi::ErrorContainer &error)
{
    const std::string name = blossomIO.input.get("name").getString();
    const std::string clusterUuid = blossomIO.input.get("cluster_uuid").getString();
    const std::string dataSetUuid = blossomIO.input.get("data_set_uuid").getString();
    const std::string taskType = blossomIO.input.get("type").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // get cluster
    Cluster* cluster = HanamiRoot::m_clusterHandler->getCluster(clusterUuid);
    if(cluster == nullptr)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "'not found";
        status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get meta-infos of data-set from shiori
    Kitsunemimi::JsonItem dataSetInfo;
    if(getDateSetInfo(dataSetInfo, dataSetUuid, context, error) == false)
    {
        error.addMeesage("Failed to get information from shiori for UUID '" + dataSetUuid + "'");
        // TODO: add status-error from response from shiori
        status.statusCode = Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE;
        return false;
    }

    // create task
    std::string taskUuid = "";
    if(dataSetInfo.get("type").getString() == "mnist")
    {
        imageTask(taskUuid,
                  name,
                  taskType,
                  dataSetUuid,
                  userContext,
                  cluster,
                  dataSetInfo,
                  status,
                  error);
    }
    else if(dataSetInfo.get("type").getString() == "csv")
    {
        tableTask(taskUuid,
                  name,
                  taskType,
                  dataSetUuid,
                  userContext,
                  cluster,
                  dataSetInfo,
                  status,
                  error);
    }
    else
    {
        status.errorMessage = "Invalid dataset-type '"
                              + dataSetInfo.get("type").getString()
                              + "' given for to create new task";
        status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // create output
    blossomIO.output.insert("uuid", taskUuid);
    blossomIO.output.insert("name", name);

    return true;
}

/**
 * @brief add image-task to queue of cluster
 *
 * @param taskUuid reference for the output of the uuid of the new task
 * @param name name of the task
 * @param taskType type of the task (learn or request)
 * @param dataSetLocation location of the base-dataset for the task
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
                      const std::string &dataSetUuid,
                      const Kitsunemimi::Hanami::UserContext &userContext,
                      Cluster* cluster,
                      Kitsunemimi::JsonItem &dataSetInfo,
                      BlossomStatus &status,
                      Kitsunemimi::ErrorContainer &error)
{
    // get input-data
    const std::string dataSetLocation = dataSetInfo.get("location").getString();
    float* floatData = getDataSetPayload(dataSetLocation, error);
    if(floatData == nullptr)
    {
        error.addMeesage("Failed to get data of data-set from location '"
                         + dataSetLocation
                         + "'");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get relevant information from output
    const uint64_t numberOfInputs = dataSetInfo.get("inputs").getLong();
    const uint64_t numberOfOutputs = dataSetInfo.get("outputs").getLong();
    const uint64_t numberOfLines = dataSetInfo.get("lines").getLong();

    if(taskType == "learn")
    {
        taskUuid = cluster->addImageLearnTask(name,
                                              userContext.userId,
                                              userContext.projectId,
                                              floatData,
                                              numberOfInputs,
                                              numberOfOutputs,
                                              numberOfLines);
    }
    else
    {
        taskUuid = cluster->addImageRequestTask(name,
                                                userContext.userId,
                                                userContext.projectId,
                                                floatData,
                                                numberOfInputs,
                                                numberOfOutputs,
                                                numberOfLines);
    }

    return true;
}

/**
 * @brief add table-task to queue of cluster
 *
 * @param taskUuid reference for the output of the uuid of the new task
 * @param name name of the task
 * @param taskType type of the task (learn or request)
 * @param dataSetLocation location of the base-dataset for the task
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
                      const std::string &dataSetLocation,
                      const Kitsunemimi::Hanami::UserContext &userContext,
                      Cluster* cluster,
                      Kitsunemimi::JsonItem &dataSetInfo,
                      BlossomStatus &status,
                      Kitsunemimi::ErrorContainer &error)
{
    // init request-task
    InputSegment* inSegment = cluster->inputSegments.begin()->second;
    OutputSegment* outSegment = cluster->outputSegments.begin()->second;
    const uint64_t numberOfInputs = inSegment->segmentHeader->inputs.count;
    const uint64_t numberOfOutputs = outSegment->segmentHeader->outputs.count;
    const uint64_t numberOfLines = dataSetInfo.get("lines").getLong();

    // get input-data
    const std::string inputColumnName = inSegment->getName();
    float* inputBuffer = getDataSetPayload(dataSetLocation, error, inputColumnName);
    if(inputBuffer == nullptr)
    {
        error.addMeesage("Failed to get data of data-set from location '"
                         + dataSetLocation
                         + "' and column with name '"
                         + inputColumnName
                         + "'");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    if(taskType == "request")
    {
        taskUuid = cluster->addTableRequestTask(name,
                                                userContext.userId,
                                                userContext.projectId,
                                                inputBuffer,
                                                numberOfInputs,
                                                numberOfOutputs,
                                                numberOfLines - numberOfInputs);

    }
    else
    {
        // get output-data
        const std::string outputColumnName = outSegment->getName();
        float* outputBuffer = getDataSetPayload(dataSetLocation, error, outputColumnName);
        if(outputBuffer == nullptr)
        {
            error.addMeesage("Failed to get data of data-set from location '"
                             + dataSetLocation
                             + "' and column with name '"
                             + outputColumnName
                             + "'");
            status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
            return false;
        }

        // create task
        const uint64_t numberOfLines = dataSetInfo.get("lines").getLong();
        taskUuid = cluster->addTableLearnTask(name,
                                              userContext.userId,
                                              userContext.projectId,
                                              inputBuffer,
                                              outputBuffer,
                                              numberOfInputs,
                                              numberOfOutputs,
                                              numberOfLines - numberOfInputs);

        // clear leftover of the buffer
        delete outputBuffer;
    }

    // clear leftover of the buffer
    delete inputBuffer;

    return true;
}
