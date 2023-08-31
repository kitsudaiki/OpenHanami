/**
 * @file        get_project.cpp
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

#include "get_project.h"

#include <hanami_root.h>
#include <database/projects_table.h>

#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 */
GetProject::GetProject()
    : Blossom("Show information of a specific registered user.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id",
                       SAKURA_STRING_TYPE,
                       true,
                       "Id of the user.");
    // column in database is limited to 256 characters size
    assert(addFieldBorder("id", 4, 256));
    assert(addFieldRegex("id", ID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id",
                        SAKURA_STRING_TYPE,
                        "ID of the new user.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new user.");
    registerOutputField("creator_id",
                        SAKURA_STRING_TYPE,
                        "Id of the creator of the user.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetProject::runTask(BlossomIO &blossomIO,
                    const Kitsunemimi::DataMap &context,
                    BlossomStatus &status,
                    Kitsunemimi::ErrorContainer &error)
{
    // check if admin
    if(context.getBoolByKey("is_admin") == false)
    {
        status.statusCode = UNAUTHORIZED_RTYPE;
        return false;
    }

    // get information from request
    const std::string projectId = blossomIO.input.get("id").getString();

    // get data from table
    if(ProjectsTable::getInstance()->getProject(blossomIO.output, projectId, error) == false)
    {
        status.errorMessage = "Project with id '" + projectId + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        return false;
    }

    return true;
}
