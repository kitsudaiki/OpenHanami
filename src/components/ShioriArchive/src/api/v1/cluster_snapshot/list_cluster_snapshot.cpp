/**
 * @file        list_cluster_snapshot.cpp
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

#include "list_cluster_snapshot.h"

#include <shiori_root.h>
#include <database/cluster_snapshot_table.h>

#include <libKitsunemimiHanamiCommon/enums.h>

using namespace Kitsunemimi::Hanami;

ListClusterSnapshot::ListClusterSnapshot()
    : Blossom("List snapshots of all visible cluster.")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("header",
                        SAKURA_ARRAY_TYPE,
                        "Array with the namings all columns of the table.");
    assert(addFieldMatch("header", new Kitsunemimi::DataValue("[\"uuid\","
                                                              "\"project_id\","
                                                              "\"owner_id\","
                                                              "\"visibility\","
                                                              "\"name\"]")));

    registerOutputField("body",
                        SAKURA_ARRAY_TYPE,
                        "Array with all rows of the table, which array arrays too.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListClusterSnapshot::runTask(BlossomIO &blossomIO,
                             const Kitsunemimi::DataMap &context,
                             BlossomStatus &status,
                             Kitsunemimi::ErrorContainer &error)
{
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // get data from table
    Kitsunemimi::TableItem table;
    if(ShioriRoot::clusterSnapshotTable->getAllClusterSnapshot(table, userContext, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    blossomIO.output.insert("header", table.getInnerHeader());
    blossomIO.output.insert("body", table.getBody());

    return true;
}
