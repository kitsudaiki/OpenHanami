/**
 * @file        show_cluster.cpp
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

#include "show_cluster.h"

#include <database/cluster_table.h>
#include <hanami_root.h>

ShowCluster::ShowCluster() : Blossom("Show information of a specific cluster.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("uuid of the cluster.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE).setComment("UUID of the cluster.");

    registerOutputField("name", SAKURA_STRING_TYPE).setComment("Name of the cluster.");

    registerOutputField("owner_id", SAKURA_STRING_TYPE)
        .setComment("ID of the user, who created the cluster.");

    registerOutputField("project_id", SAKURA_STRING_TYPE)
        .setComment("ID of the project, where the cluster belongs to.");

    registerOutputField("visibility", SAKURA_STRING_TYPE)
        .setComment("Visibility of the cluster (private, shared, public).");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ShowCluster::runTask(BlossomIO& blossomIO,
                     const json& context,
                     BlossomStatus& status,
                     Hanami::ErrorContainer& error)
{
    const Hanami::UserContext userContext = convertContext(context);
    const std::string clusterUuid = blossomIO.input["uuid"];

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

    return true;
}
