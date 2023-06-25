/**
 * @file        get_progress_data_set.cpp
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

#include "get_progress_data_set.h"

#include <hanami_root.h>
#include <database/data_set_table.h>
#include <core/data_set_files/data_set_file.h>
#include <core/data_set_files/image_data_set_file.h>
#include <core/data_set_files/table_data_set_file.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>

GetProgressDataSet::GetProgressDataSet()
    : Blossom("Get upload progress of a specific data-set.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the dataset set to delete.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the data-set.");
    registerOutputField("temp_files",
                        SAKURA_MAP_TYPE,
                        "Map with the uuids of the temporary files and it's upload progress");
    registerOutputField("complete",
                        SAKURA_BOOL_TYPE,
                        "True, if all temporary files for complete.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetProgressDataSet::runTask(BlossomIO &blossomIO,
                            const Kitsunemimi::DataMap &context,
                            BlossomStatus &status,
                            Kitsunemimi::ErrorContainer &error)
{
    const std::string dataUuid = blossomIO.input.get("uuid").getString();
    const UserContext userContext(context);

    Kitsunemimi::JsonItem databaseOutput;
    if(HanamiRoot::dataSetTable->getDataSet(databaseOutput,
                                            dataUuid,
                                            userContext,
                                            error,
                                            true) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // add uuid
    blossomIO.output.insert("uuid", databaseOutput.get("uuid"));

    // parse and add temp-file-information
    const std::string tempFilesStr = databaseOutput.get("temp_files").toString();
    Kitsunemimi::JsonItem tempFiles;
    if(tempFiles.parse(tempFilesStr, error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    blossomIO.output.insert("temp_files", tempFiles);

    // check and add if complete
    const std::vector<std::string> keys = tempFiles.getKeys();
    bool finishedAll = true;
    for(uint32_t i = 0; i < keys.size(); i++)
    {
        if(tempFiles.get(keys.at(i)).getFloat() < 1.0f) {
            finishedAll = false;
        }
    }
    blossomIO.output.insert("complete", finishedAll);

    return true;
}
