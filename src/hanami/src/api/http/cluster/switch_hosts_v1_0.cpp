/**
 * @file        switch_host_v1_0.cpp
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

#include "switch_hosts_v1_0.h"

#include <core/cluster/cluster_handler.h>
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/cuda/cuda_host.h>
#include <core/processing/logical_host.h>
#include <core/processing/physical_host.h>
#include <database/cluster_table.h>
#include <hanami_root.h>

SwitchHostsV1M0::SwitchHostsV1M0()
    : Blossom("Switch the host, where the cluster should be processed.")
{
    errorCodes.push_back(CONFLICT_RTYPE);
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("cluster_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the cluster to move.")
        .setRegex(UUID_REGEX);

    registerInputField("hexagon_id", SAKURA_INT_TYPE)
        .setComment("ID of the hexagon within the cluster.")
        .setLimit(0, 1000000000);

    registerInputField("host_uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the target-host where the cluster should moved to.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("created_at", SAKURA_STRING_TYPE)
        .setComment("Timestamp, when cluster was created.");

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the moved cluster.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the moved cluster.");

    registerOutputField("owner_id", SAKURA_STRING_TYPE)
        .setComment("ID of the user, who created the cluster.");

    registerOutputField("project_id", SAKURA_STRING_TYPE)
        .setComment("ID of the project, where the cluster belongs to.");

    registerOutputField("visibility", SAKURA_STRING_TYPE)
        .setComment("Visibility of the new created cluster (private, shared, public).");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
SwitchHostsV1M0::runTask(BlossomIO& blossomIO,
                         const Hanami::UserContext& userContext,
                         BlossomStatus& status,
                         Hanami::ErrorContainer& error)
{
    const std::string clusterUuid = blossomIO.input["cluster_uuid"];
    const std::string hostUuid = blossomIO.input["host_uuid"];
    const uint64_t hexagonId = blossomIO.input["hexagon_id"];

    // get data from table
    json clusterResult;
    ReturnStatus ret = ClusterTable::getInstance()->getCluster(
        blossomIO.output, clusterUuid, userContext, false, error);
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

    // get target-host
    LogicalHost* targetHost = HanamiRoot::physicalHost->getHost(hostUuid);
    if (targetHost == nullptr) {
        status.errorMessage = "Host with uuid '" + hostUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get cluster
    Cluster* cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if (cluster == nullptr) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMessage("Cluster with UUID '" + clusterUuid
                         + "' not found even it exists within the database");
        return false;
    }

    // check if hexagon-id even exist within the cluster
    if (cluster->hexagons.size() <= hexagonId) {
        status.errorMessage = "Cluster with UUID '" + clusterUuid
                              + "' does not contain a hexagon with ID '" + std::to_string(hexagonId)
                              + "'";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // move cluster to another host
    if (targetHost->moveHexagon(&cluster->hexagons[hexagonId]) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMessage("Failed to move cluster with UUID '" + clusterUuid
                         + "' to host with UUID '" + hostUuid + "'");
        return false;
    }

    return true;
}
