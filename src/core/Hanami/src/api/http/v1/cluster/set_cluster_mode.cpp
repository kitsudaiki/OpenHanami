/**
 * @file        set_cluster_mode.cpp
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

#include "set_cluster_mode.h"

#include <hanami_root.h>
#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>

SetClusterMode::SetClusterMode()
    : Blossom("Set mode of the cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster.");
    assert(addFieldRegex("uuid", UUID_REGEX));
    registerInputField("connection_uuid",
                       SAKURA_STRING_TYPE,
                       false,
                       "UUID of the connection for input and output.");
    assert(addFieldRegex("connection_uuid", UUID_REGEX));
    registerInputField("new_state",
                       SAKURA_STRING_TYPE,
                       true,
                       "New desired state for the cluster.");
    assert(addFieldRegex("new_state", "^(TASK|DIRECT)$"));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the cluster.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the cluster.");
    registerOutputField("new_state",
                        SAKURA_STRING_TYPE,
                        "New desired state for the cluster.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
SetClusterMode::runTask(BlossomIO &blossomIO,
                        const Kitsunemimi::DataMap &context,
                        BlossomStatus &status,
                        Kitsunemimi::ErrorContainer &error)
{
    const std::string clusterUuid = blossomIO.input.get("uuid").getString();
    const std::string connectionUuid = blossomIO.input.get("connection_uuid").getString();
    const std::string newState = blossomIO.input.get("new_state").getString();
    const UserContext userContext(context);

    // get data from table
    if(ClusterTable::getInstance()->getCluster(blossomIO.output,
                                               clusterUuid,
                                               userContext,
                                               error) == false)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get cluster
    Cluster* cluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
    if(cluster == nullptr)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "'not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // switch mode of cluster
    if(cluster->setClusterState(newState) == false)
    {
        status.errorMessage = "Can not switch Cluster with uuid '"
                              + clusterUuid
                              + "' to new mode '"
                              + newState
                              + "'";
        // TODO: get exact reason, why it was not successful
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // handle client
    if(newState == "DIRECT")
    {
        // get internal connection
        /*HanamiMessaging* messageing = HanamiMessaging::getInstance();
        HanamiMessagingClient* client =  messageing->getIncomingClient(connectionUuid);
        if(client == nullptr)
        {
            status.errorMessage = "Connection with UUID '" + connectionUuid + "'not found";
            status.statusCode = NOT_FOUND_RTYPE;
            error.addMeesage(status.errorMessage);
            return false;
        }
        client->setStreamCallback(cluster, streamDataCallback);
        cluster->msgClient = client;*/
    }
    else
    {
        //
    }

    blossomIO.output.insert("new_state", newState);

    // remove irrelevant fields
    blossomIO.output.remove("owner_id");
    blossomIO.output.remove("project_id");
    blossomIO.output.remove("visibility");

    return true;
}
