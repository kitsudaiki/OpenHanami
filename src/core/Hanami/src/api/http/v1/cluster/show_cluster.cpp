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

#include <hanami_root.h>

#include <libKitsunemimiJson/json_item.h>

ShowCluster::ShowCluster()
    : Blossom("Show information of a specific cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "uuid of the cluster.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the cluster.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the cluster.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the cluster.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the cluster belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the cluster (private, shared, public).");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ShowCluster::runTask(BlossomIO &blossomIO,
                     const Kitsunemimi::DataMap &context,
                     BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);
    const std::string clusterUuid = blossomIO.input.get("uuid").getString();

    // get data from table
    if(HanamiRoot::clustersTable->getCluster(blossomIO.output,
                                             clusterUuid,
                                             userContext,
                                             error) == false)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    return true;
}
