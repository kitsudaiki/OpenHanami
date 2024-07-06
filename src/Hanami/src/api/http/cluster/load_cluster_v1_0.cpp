/**
 * @file        load_cluster_v1_0.cpp
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

#include "load_cluster_v1_0.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/statemachine_init.h>
#include <database/checkpoint_table.h>
#include <database/cluster_table.h>
#include <hanami_common/statemachine.h>
#include <hanami_root.h>

LoadClusterV1M0::LoadClusterV1M0()
    : Blossom("Load a checkpoint from shiori into an existing cluster and override the old data.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the cluster, where the checkpoint should be loaded into.")
        .setRegex(UUID_REGEX);

    registerInputField("checkpoint_uuid", SAKURA_STRING_TYPE)
        .setComment(
            "UUID of the checkpoint, which should be loaded from shiori "
            "into the cluster.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the load-task in the queue of the cluster.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
LoadClusterV1M0::runTask(BlossomIO& blossomIO,
                         const json& context,
                         BlossomStatus& status,
                         Hanami::ErrorContainer& error)
{
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const std::string checkpointUuid = blossomIO.input["checkpoint_uuid"];
    const Hanami::UserContext userContext = convertContext(context);

    // get data from table
    ClusterTable::ClusterDbEntry clusterInfo;
    ReturnStatus ret
        = ClusterTable::getInstance()->getCluster(clusterInfo, clusterUuid, userContext, error);
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

    // get meta-infos of dataset from shiori
    CheckpointTable::CheckpointDbEntry checkpointInfo;
    ret = CheckpointTable::getInstance()->getCheckpoint(
        checkpointInfo, checkpointUuid, userContext, error);
    if (ret == ERROR) {
        error.addMessage("Failed to get information from database for UUID '" + checkpointUuid
                         + "'");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Checkpoint with uuid '" + checkpointUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // create new task
    Task* newTask = cluster->addNewTask();
    if (newTask == nullptr) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    newTask->name = "";
    newTask->userId = userContext.userId;
    newTask->projectId = userContext.projectId;
    newTask->type = CLUSTER_CHECKPOINT_RESTORE_TASK;
    newTask->progress.queuedTimeStamp = std::chrono::system_clock::now();
    newTask->progress.totalNumberOfCycles = 1;

    // fill metadata
    CheckpointRestoreInfo info;
    info.checkpointInfo = checkpointInfo;
    newTask->info = std::move(info);

    cluster->stateMachine->goToNextState(PROCESS_TASK);

    blossomIO.output["uuid"] = newTask->uuid.toString();

    return true;
}
