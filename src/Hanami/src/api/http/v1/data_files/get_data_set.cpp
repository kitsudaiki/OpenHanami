/**
 * @file        get_data_set.cpp
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

#include "get_data_set.h"

#include <hanami_root.h>
#include <database/data_set_table.h>

#include <hanami_files/data_set_files/data_set_functions.h>

#include <hanami_json/json_item.h>

GetDataSet::GetDataSet()
    : Blossom("Get information of a specific data-set.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the data-set set to delete.")
            .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the data-set.");

    registerOutputField("name", SAKURA_STRING_TYPE)
            .setComment("Name of the data-set.");

    registerOutputField("owner_id", SAKURA_STRING_TYPE)
            .setComment("ID of the user, who created the data-set.");

    registerOutputField("project_id", SAKURA_STRING_TYPE)
            .setComment("ID of the project, where the data-set belongs to.");

    registerOutputField("visibility", SAKURA_STRING_TYPE)
            .setComment("Visibility of the data-set (private, shared, public).");

    registerOutputField("location", SAKURA_STRING_TYPE)
            .setComment("Local file-path of the data-set.");

    registerOutputField("type", SAKURA_STRING_TYPE)
            .setComment("Type of the new set (csv or mnist)");

    registerOutputField("inputs", SAKURA_INT_TYPE)
            .setComment("Number of inputs.");

    registerOutputField("outputs", SAKURA_INT_TYPE)
            .setComment("Number of outputs.");

    registerOutputField("lines", SAKURA_INT_TYPE)
            .setComment("Number of lines.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetDataSet::runTask(BlossomIO &blossomIO,
                      const Hanami::DataMap &context,
                      BlossomStatus &status,
                      Hanami::ErrorContainer &error)
{
    const std::string dataUuid = blossomIO.input.get("uuid").getString();
    if(DataSetTable::getInstance()->getDateSetInfo(blossomIO.output,
                                                   dataUuid,
                                                   context,
                                                   error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // handle not found
    if(blossomIO.output.size() == 0)
    {
        status.errorMessage = "Data-set with uuid '" + dataUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    return true;
}
