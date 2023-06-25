/**
 * @file        create_project.cpp
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

#include "create_project.h"

#include <hanami_root.h>

#include <libKitsunemimiCrypto/hashes.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJson/json_item.h>

/**
 * @brief constructor
 */
CreateProject::CreateProject()
    : Blossom("Register a new project within Misaki.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the new project.");
    // column in database is limited to 256 characters size
    assert(addFieldBorder("id", 4, 256));
    assert(addFieldRegex("id", ID_REGEX));

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name of the new project.");
    // column in database is limited to 256 characters size
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", NAME_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("id",
                        SAKURA_STRING_TYPE,
                        "ID of the new project.");

    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new project.");

    registerOutputField("creator_id",
                        SAKURA_STRING_TYPE,
                        "Id of the creator of the project.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateProject::runTask(BlossomIO &blossomIO,
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
    const std::string projectName = blossomIO.input.get("name").getString();
    const std::string creatorId = context.getStringByKey("id");

    // check if user already exist within the table
    Kitsunemimi::JsonItem getResult;
    if(HanamiRoot::projectsTable->getProject(getResult, projectId, error))
    {
        status.errorMessage = "Project with id '" + projectId + "' already exist.";
        status.statusCode = CONFLICT_RTYPE;
        return false;
    }

    // convert values
    Kitsunemimi::JsonItem userData;
    userData.insert("id", projectId);
    userData.insert("name", projectName);
    userData.insert("creator_id", creatorId);

    // add new user to table
    if(HanamiRoot::projectsTable->addProject(userData, error) == false)
    {
        status.errorMessage = error.toString();
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get new created user from database
    if(HanamiRoot::projectsTable->getProject(blossomIO.output,
                                                   projectId,
                                                   error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
