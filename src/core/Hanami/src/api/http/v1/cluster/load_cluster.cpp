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
#include <database/cluster_snapshot_table.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>

LoadCluster::LoadCluster()
    : Blossom("Load a snapshot from shiori into an existing cluster and override the old data.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("cluster_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster, where the snapshot should be loaded into.");
    assert(addFieldRegex("cluster_uuid", UUID_REGEX));

    registerInputField("snapshot_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the snapshot, which should be loaded from shiori "
                       "into the cluster.");
    assert(addFieldRegex("snapshot_uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the load-task in the queue of the cluster.");

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
    const std::string snapshotUuid = blossomIO.input.get("snapshot_uuid").getString();
    const UserContext userContext(context);

    // get cluster
    Cluster* cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if(cluster == nullptr)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get meta-infos of data-set from shiori
    Kitsunemimi::JsonItem parsedSnapshotInfo;
    if(ClusterSnapshotTable::getInstance()->getClusterSnapshot(parsedSnapshotInfo,
                                                               snapshotUuid,
                                                               userContext,
                                                               error,
                                                               true) == false)
    {
        error.addMeesage("Failed to get information from database for UUID '" + snapshotUuid + "'");
        status.statusCode = NOT_FOUND_RTYPE;
        return false;
    }

    // init request-task
    const std::string infoStr = parsedSnapshotInfo.toString();
    const std::string taskUuid = cluster->addClusterSnapshotRestoreTask("",
                                                                        infoStr,
                                                                        userContext.userId,
                                                                        userContext.projectId);
    blossomIO.output.insert("uuid", taskUuid);

    return true;
}
