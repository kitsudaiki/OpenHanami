/**
 * @file        show_task.cpp
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

#include "show_task.h"

#include <core/cluster/cluster_handler.h>
#include <core/cluster/cluster.h>
#include <hanami_root.h>

ShowTask::ShowTask()
    : Blossom("Show information of a specific task.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster, which should process the request");
    assert(addFieldRegex("uuid", UUID_REGEX));

    registerInputField("cluster_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster, which should process the request");
    assert(addFieldRegex("cluster_uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("percentage_finished",
                        SAKURA_FLOAT_TYPE,
                        "Percentation of the progress between 0.0 and 1.0.");
    registerOutputField("state",
                        SAKURA_STRING_TYPE,
                        "Actual state of the task (queued, active, aborted or finished).");
    registerOutputField("queue_timestamp",
                        SAKURA_STRING_TYPE,
                        "Timestamp in UTC when the task entered the queued state, "
                        "which is basicall the timestamp when the task was created");
    registerOutputField("start_timestamp",
                        SAKURA_STRING_TYPE,
                        "Timestamp in UTC when the task entered the active state.");
    registerOutputField("end_timestamp",
                        SAKURA_STRING_TYPE,
                        "Timestamp in UTC when the task was finished.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}


/**
 * @brief runTask
 */
bool
ShowTask::runTask(BlossomIO &blossomIO,
                  const Kitsunemimi::DataMap &context,
                  BlossomStatus &status,
                  Kitsunemimi::ErrorContainer &error)
{
    const std::string clusterUuid = blossomIO.input.get("cluster_uuid").getString();
    const std::string taskUuid = blossomIO.input.get("uuid").getString();
    const UserContext userContext(context);

    // get cluster
    Cluster* cluster = HanamiRoot::m_clusterHandler->getCluster(clusterUuid);
    if(cluster == nullptr)
    {
        status.errorMessage = "Cluster with UUID '" + clusterUuid + "'not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    const TaskProgress progress = cluster->getProgress(taskUuid);

    // get basic information
    blossomIO.output.insert("percentage_finished", progress.percentageFinished);
    blossomIO.output.insert("queue_timestamp", serializeTimePoint(progress.queuedTimeStamp));

    // get timestamps
    if(progress.state == QUEUED_TASK_STATE)
    {
        blossomIO.output.insert("state", "queued");
        blossomIO.output.insert("start_timestamp", "-");
        blossomIO.output.insert("end_timestamp", "-");
    }
    else if(progress.state == ACTIVE_TASK_STATE)
    {
        blossomIO.output.insert("state", "active");
        blossomIO.output.insert("start_timestamp",
                                  serializeTimePoint(progress.startActiveTimeStamp));
        blossomIO.output.insert("end_timestamp", "-");
    }
    else if(progress.state == ABORTED_TASK_STATE)
    {
        blossomIO.output.insert("state", "aborted");
        blossomIO.output.insert("start_timestamp",
                                  serializeTimePoint(progress.startActiveTimeStamp));
        blossomIO.output.insert("end_timestamp", "-");
    }
    else if(progress.state == FINISHED_TASK_STATE)
    {
        blossomIO.output.insert("state", "finished");
        blossomIO.output.insert("start_timestamp",
                                  serializeTimePoint(progress.startActiveTimeStamp));
        blossomIO.output.insert("end_timestamp", serializeTimePoint(progress.endActiveTimeStamp));
    }

    return true;
}
