/**
 * @file        create_checkpoint.cpp
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

#include "create_checkpoint.h"

#include <hanami_root.h>
#include <database/checkpoint_table.h>
#include <core/temp_file_handler.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>

CreateCheckpoint::CreateCheckpoint()
    : Blossom("Init new checkpoint of a cluster.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    // HINT(kitsudaiki): Checkpoints are created internally asynchrous by the cluster and get the same
    //                   uuid as the task, where the checkpoint was created. Because of this the uuid
    //                   has to be predefined.
    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the new checkpoint.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name of the new checkpoint.");
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", NAME_REGEX));

    registerInputField("user_id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the user, who owns the checkpoint.");
    assert(addFieldBorder("user_id", 4, 256));
    assert(addFieldRegex("user_id", ID_EXT_REGEX));

    registerInputField("project_id",
                       SAKURA_STRING_TYPE,
                       true,
                       "ID of the project, where the checkpoint belongs to.");
    assert(addFieldBorder("project_id", 4, 256));
    assert(addFieldRegex("project_id", ID_REGEX));

    registerInputField("header",
                       SAKURA_MAP_TYPE,
                       true,
                       "Header of the file with information of the splits.");

    registerInputField("input_data_size",
                       SAKURA_INT_TYPE,
                       true,
                       "Total size of the checkpoint in number of bytes.");
    assert(addFieldBorder("input_data_size", 1, 10000000000));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new checkpoint.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new checkpoint.");
    registerOutputField("uuid_input_file",
                        SAKURA_STRING_TYPE,
                        "UUID to identify the file for data upload of the checkpoint.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CreateCheckpoint::runTask(BlossomIO &blossomIO,
                               const Kitsunemimi::DataMap &,
                               BlossomStatus &status,
                               Kitsunemimi::ErrorContainer &error)
{
    const std::string uuid = blossomIO.input.get("uuid").getString();
    const std::string name = blossomIO.input.get("name").getString();
    const std::string userId = blossomIO.input.get("id").getString();
    const std::string projectId = blossomIO.input.get("project_id").getString();
    const long inputDataSize = blossomIO.input.get("input_data_size").getLong();

    // checkpoints are created by another internal process, which gives the id's not in the context
    // object, but as normal values
    UserContext userContext;
    userContext.userId = userId;
    userContext.projectId = projectId;

    // get directory to store data from config
    bool success = false;
    std::string targetFilePath = GET_STRING_CONFIG("storage", "checkpoint_location", success);
    if(success == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("checkpoint-location to store checkpoint is missing in the config");
        return false;
    }

    // init temp-file for input-data
    const std::string tempFileUuid = generateUuid().toString();
    if(TempFileHandler::getInstance()->initNewFile(tempFileUuid, inputDataSize) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to initialize temporary file for new input-data.");
        return false;
    }    

    // build absolut file-path to store the file
    if(targetFilePath.at(targetFilePath.size() - 1) != '/') {
        targetFilePath.append("/");
    }
    targetFilePath.append(uuid + "_checkpoint_" + userId);

    // register in database
    blossomIO.output.insert("uuid", uuid);
    blossomIO.output.insert("name", name);
    blossomIO.output.insert("location", targetFilePath);
    blossomIO.output.insert("header", blossomIO.input.get("header"));
    blossomIO.output.insert("project_id", projectId);
    blossomIO.output.insert("owner_id", userId);
    blossomIO.output.insert("visibility", "private");

    // init placeholder for temp-file progress to database
    Kitsunemimi::JsonItem tempFiles;
    tempFiles.insert(tempFileUuid, Kitsunemimi::JsonItem(0.0f));
    blossomIO.output.insert("temp_files", tempFiles);

    // add to database
    if(CheckpointTable::getInstance()->addCheckpoint(blossomIO.output,
                                                               userContext,
                                                               error) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // add values to output
    blossomIO.output.insert("uuid_input_file", tempFileUuid);

    // remove blocked values from output
    blossomIO.output.remove("location");
    blossomIO.output.remove("header");
    blossomIO.output.remove("project_id");
    blossomIO.output.remove("owner_id");
    blossomIO.output.remove("visibility");
    blossomIO.output.remove("temp_files");

    return true;
}
