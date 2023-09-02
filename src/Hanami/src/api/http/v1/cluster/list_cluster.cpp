/**
 * @file        list_cluster.cpp
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

#include "list_cluster.h"

#include <hanami_root.h>
#include <database/cluster_table.h>

ListCluster::ListCluster()
    : Blossom("List all visible clusters.")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("header", SAKURA_ARRAY_TYPE)
            .setComment("Array with the names all columns of the table.")
            .setMatch(new Hanami::DataValue("[\"uuid\","
                                                 "\"project_id\","
                                                 "\"owner_id\","
                                                 "\"visibility\","
                                                 "\"name\"]"));

    registerOutputField("body", SAKURA_ARRAY_TYPE)
            .setComment("Json-string with all information of all vilible clusters.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListCluster::runTask(BlossomIO &blossomIO,
                     const Hanami::DataMap &context,
                     BlossomStatus &status,
                     Hanami::ErrorContainer &error)
{
    const UserContext userContext(context);

    // get data from table
    Hanami::TableItem table;
    if(ClusterTable::getInstance()->getAllCluster(table, userContext, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to get all clusters form database");
        return false;
    }

    // create output
    Hanami::DataArray* headerInfo = table.getInnerHeader();
    blossomIO.output.insert("header", headerInfo);
    blossomIO.output.insert("body", table.getBody());
    delete headerInfo;

    return true;
}
