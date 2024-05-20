/**
 * @file        list_hosts.cpp
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

#include "list_hosts.h"

#include <core/processing/physical_host.h>
#include <hanami_root.h>

ListHosts::ListHosts() : Blossom("List all logical hosts.")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    json headerMatch = json::array();
    headerMatch.push_back("uuid");
    headerMatch.push_back("type");

    registerOutputField("header", SAKURA_ARRAY_TYPE)
        .setComment("Array with the names all columns of the table.")
        .setMatch(headerMatch);

    registerOutputField("body", SAKURA_ARRAY_TYPE)
        .setComment("Json-string with all information of all vilible hosts.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ListHosts::runTask(BlossomIO& blossomIO,
                   const json& context,
                   BlossomStatus& status,
                   Hanami::ErrorContainer& error)
{
    const Hanami::UserContext userContext = convertContext(context);

    // prepare header
    json header = json::array();
    header.push_back("uuid");
    header.push_back("type");

    // get data from table
    const json body = HanamiRoot::physicalHost->getAllHostsAsJson();

    // create output
    blossomIO.output["header"] = header;
    blossomIO.output["body"] = body;

    return true;
}
