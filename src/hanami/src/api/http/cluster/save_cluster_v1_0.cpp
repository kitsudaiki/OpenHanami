/**
 * @file        save_cluster_v1_0.cpp
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

#include "save_cluster_v1_0.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/statemachine_init.h>
#include <database/cluster_table.h>
#include <hanami_common/statemachine.h>
#include <hanami_root.h>

SaveClusterV1M0::SaveClusterV1M0() : Blossom("Save a cluster.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name", SAKURA_STRING_TYPE)
        .setComment("Name for task, which is place in the task-queue and for the new checkpoint.")
        .setLimit(4, 254)
        .setRegex(NAME_REGEX);

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the cluster, which should be saved as new snapstho to shiori.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the save-task in the queue of the cluster.");

    registerOutputField("name", SAKURA_STRING_TYPE)
        .setComment(
            "Name of the new created task and of the checkpoint, "
            "which should be created by the task.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
SaveClusterV1M0::runTask(BlossomIO& blossomIO,
                         const Hanami::UserContext& userContext,
                         BlossomStatus& status,
                         Hanami::ErrorContainer& error)
{
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const std::string name = blossomIO.input["name"];

    // get data from table
    json clusterResult;
    ReturnStatus ret = ClusterTable::getInstance()->getCluster(
        clusterResult, clusterUuid, userContext, false, error);
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
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMessage("Cluster with UUID '" + clusterUuid
                         + "'not found even it exists within the database");
        return false;
    }

    // create new task
    Task* newTask = cluster->addNewTask();
    if (newTask == nullptr) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    newTask->name = name;
    newTask->userId = userContext.userId;
    newTask->projectId = userContext.projectId;
    newTask->type = CLUSTER_CHECKPOINT_SAVE_TASK;
    newTask->progress.queuedTimeStamp = std::chrono::system_clock::now();
    newTask->progress.totalNumberOfCycles = 1;

    // fill metadata
    CheckpointSaveInfo info;
    info.checkpointName = name;
    newTask->info = std::move(info);

    cluster->stateMachine->goToNextState(PROCESS_TASK);

    blossomIO.output["uuid"] = newTask->uuid.toString();
    blossomIO.output["name"] = name;

    return true;
}
