/**
 * @file        list_dataset.cpp
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

#include "list_dataset.h"

#include <database/dataset_table.h>
#include <hanami_root.h>

ListDataSet::ListDataSet() : Blossom("List all visible datasets.")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    json headerMatch = json::array();
    headerMatch.push_back("created_at");
    headerMatch.push_back("uuid");
    headerMatch.push_back("project_id");
    headerMatch.push_back("owner_id");
    headerMatch.push_back("visibility");
    headerMatch.push_back("name");
    headerMatch.push_back("type");

    registerOutputField("header", SAKURA_ARRAY_TYPE)
        .setComment("Array with the namings all columns of the table.")
        .setMatch(headerMatch);

    registerOutputField("body", SAKURA_ARRAY_TYPE)
        .setComment("Array with all rows of the table, which array arrays too.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListDataSet::runTask(BlossomIO& blossomIO,
                     const json& context,
                     BlossomStatus& status,
                     Hanami::ErrorContainer& error)
{
    const Hanami::UserContext userContext = convertContext(context);

    // get data from table
    Hanami::TableItem table;
    if (DataSetTable::getInstance()->getAllDataSet(table, userContext, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // create output
    blossomIO.output["header"] = table.getInnerHeader();
    blossomIO.output["body"] = table.getBody();

    return true;
}
