/**
 * @file        create_cluster_template.h
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

#include "show_template.h"

#include <libKitsunemimiCrypto/common.h>

#include <hanami_root.h>

ShowTemplate::ShowTemplate()
    : Blossom("Show a specific template.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the template.");
    assert(addFieldRegex("uuid", "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-"
                                 "[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the template.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the template.");
    registerOutputField("template",
                        SAKURA_STRING_TYPE,
                        "The template itself.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the template.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the template belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the template (private, shared, public).");


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ShowTemplate::runTask(BlossomIO &blossomIO,
                      const Kitsunemimi::DataMap &context,
                      BlossomStatus &status,
                      Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);
    const std::string uuid = blossomIO.input.get("uuid").getString();
    // TODO: check type-field

    // get data from table
    if(HanamiRoot::templateTable->getTemplate(blossomIO.output,
                                              uuid,
                                              userContext,
                                              error,
                                              true) == false)
    {
        status.errorMessage = "Tempalte with UUID '" + uuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    blossomIO.output.insert("template", blossomIO.output.get("data").getString());

    // remove irrelevant fields
    blossomIO.output.remove("data");

    return true;
}
