/**
 * @file        create_csv_data_set.cpp
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

#include "create_csv_data_set.h"

#include <shiori_root.h>
#include <database/data_set_table.h>
#include <core/temp_file_handler.h>

#include <libKitsunemimiHanamiCommon/uuid.h>
#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiCommon/defines.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>

using namespace Kitsunemimi::Hanami;

CreateCsvDataSet::CreateCsvDataSet()
    : Blossom("Init new csv-file data-set.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name of the new data-set.");
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", NAME_REGEX));

    registerInputField("input_data_size",
                       SAKURA_INT_TYPE,
                       true,
                       "Total size of the input-data.");
    assert(addFieldBorder("input_data_size", 1, 10000000000));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new data-set.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new data-set.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the data-set.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the data-set belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the data-set (private, shared, public).");
    registerOutputField("type",
                        SAKURA_STRING_TYPE,
                        "Type of the new set (csv)");
    registerOutputField("uuid_input_file",
                        SAKURA_STRING_TYPE,
                        "UUID to identify the file for date upload of input-data.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

bool
CreateCsvDataSet::runTask(BlossomIO &blossomIO,
                          const Kitsunemimi::DataMap &context,
                          BlossomStatus &status,
                          Kitsunemimi::ErrorContainer &error)
{
    const std::string name = blossomIO.input.get("name").getString();
    const long inputDataSize = blossomIO.input.get("input_data_size").getLong();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // get directory to store data from config
    bool success = false;
    std::string targetFilePath = GET_STRING_CONFIG("shiori", "data_set_location", success);
    if(success == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("file-location to store dataset is missing in the config");
        return false;
    }

    // init temp-file for input-data
    const std::string inputUuid = Kitsunemimi::Hanami::generateUuid().toString();
    if(ShioriRoot::tempFileHandler->initNewFile(inputUuid, inputDataSize) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to initialize temporary file for new input-data.");
        return false;
    }

    // build absolut file-path to store the file
    if(targetFilePath.at(targetFilePath.size() - 1) != '/') {
        targetFilePath.append("/");
    }
    targetFilePath.append(name + "_csv_" +userContext. userId);

    // register in database
    blossomIO.output.insert("name", name);
    blossomIO.output.insert("type", "csv");
    blossomIO.output.insert("location", targetFilePath);
    blossomIO.output.insert("project_id", userContext.projectId);
    blossomIO.output.insert("owner_id", userContext.userId);
    blossomIO.output.insert("visibility", "private");

    // init placeholder for temp-file progress to database
    Kitsunemimi::JsonItem tempFiles;
    tempFiles.insert(inputUuid, Kitsunemimi::JsonItem(0.0f));
    blossomIO.output.insert("temp_files", tempFiles);

    // add to database
    if(ShioriRoot::dataSetTable->addDataSet(blossomIO.output, userContext, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // add values to output
    blossomIO.output.insert("uuid_input_file", inputUuid);

    // remove blocked values from output
    blossomIO.output.remove("location");
    blossomIO.output.remove("temp_files");

    return true;
}
