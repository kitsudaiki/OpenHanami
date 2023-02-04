/**
 * @file        get_cluster_snapshot.cpp
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

#include "get_cluster_snapshot.h"

#include <shiori_root.h>
#include <database/cluster_snapshot_table.h>

#include <libKitsunemimiHanamiCommon/enums.h>

using namespace Kitsunemimi::Hanami;

GetClusterSnapshot::GetClusterSnapshot()
    : Blossom("Get snapshot of a cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the original request-task, which placed the result in shiori.");
    assert(addFieldRegex("uuid", "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-"
                                 "[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the data-set.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the data-set.");
    registerOutputField("location",
                        SAKURA_STRING_TYPE,
                        "File path on local storage.");
    registerOutputField("header",
                        SAKURA_MAP_TYPE,
                        "Header-information of the snapshot-file.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetClusterSnapshot::runTask(BlossomIO &blossomIO,
                            const Kitsunemimi::DataMap &context,
                            BlossomStatus &status,
                            Kitsunemimi::ErrorContainer &error)
{
    const std::string dataUuid = blossomIO.input.get("uuid").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    if(ShioriRoot::clusterSnapshotTable->getClusterSnapshot(blossomIO.output,
                                                            dataUuid,
                                                            userContext,
                                                            error,
                                                            true) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // remove irrelevant fields
    blossomIO.output.remove("owner_id");
    blossomIO.output.remove("project_id");
    blossomIO.output.remove("visibility");
    blossomIO.output.remove("temp_files");

    return true;
}
