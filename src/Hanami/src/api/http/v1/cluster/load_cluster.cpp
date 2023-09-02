/**
 * @file        load_cluster.cpp
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

#include "load_cluster.h"

#include <hanami_root.h>
#include <database/checkpoint_table.h>
#include <database/cluster_table.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>
#include <core/cluster/add_tasks.h>

LoadCluster::LoadCluster()
    : Blossom("Load a checkpoint from shiori into an existing cluster and override the old data.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the cluster, where the checkpoint should be loaded into.")
            .setRegex(UUID_REGEX);

    registerInputField("checkpoint_uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the checkpoint, which should be loaded from shiori "
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
LoadCluster::runTask(BlossomIO &blossomIO,
                     const Kitsunemimi::DataMap &context,
                     BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error)
{
    const std::string clusterUuid = blossomIO.input.get("cluster_uuid").getString();
    const std::string checkpointUuid = blossomIO.input.get("checkpoint_uuid").getString();
    const UserContext userContext(context);

    // get data from table
    Kitsunemimi::JsonItem clusterInfo;
    if(ClusterTable::getInstance()->getCluster(clusterInfo,
                                               clusterUuid,
                                               userContext,
                                               error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(clusterInfo.size() == 0)
    {
        status.errorMessage = "Cluster with uuid '" + clusterUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get cluster
    Cluster* cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if(cluster == nullptr)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Cluster with UUID '"
                         + clusterUuid
                         + "'not found even it exists within the database");
        return false;
    }

    // get meta-infos of data-set from shiori
    Kitsunemimi::JsonItem parsedCheckpointInfo;
    if(CheckpointTable::getInstance()->getCheckpoint(parsedCheckpointInfo,
                                                     checkpointUuid,
                                                     userContext,
                                                     error,
                                                     true) == false)
    {
        error.addMeesage("Failed to get information from database for UUID '" + checkpointUuid + "'");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(clusterInfo.size() == 0)
    {
        status.errorMessage = "Checkpoint with uuid '" + checkpointUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // init request-task
    const std::string infoStr = parsedCheckpointInfo.toString();
    const std::string taskUuid = addCheckpointRestoreTask(*cluster,
                                                          "",
                                                          infoStr,
                                                          userContext.userId,
                                                          userContext.projectId);
    blossomIO.output.insert("uuid", taskUuid);

    return true;
}
