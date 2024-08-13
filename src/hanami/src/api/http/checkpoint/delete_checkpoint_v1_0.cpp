/**
 * @file        delete_checkpoint_v1_0.cpp
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

#include "delete_checkpoint_v1_0.h"

#include <database/checkpoint_table.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_root.h>

DeleteCheckpointV1M0::DeleteCheckpointV1M0() : Blossom("Delete a result-set from shiori.")
{
    errorCodes.push_back(NOT_FOUND_RTYPE);

    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
        .setComment("UUID of the checkpoint to delete.")
        .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteCheckpointV1M0::runTask(BlossomIO& blossomIO,
                              const Hanami::UserContext& userContext,
                              BlossomStatus& status,
                              Hanami::ErrorContainer& error)
{
    const std::string checkpointUuid = blossomIO.input["uuid"];

    // get location from database
    CheckpointTable::CheckpointDbEntry result;
    ReturnStatus ret
        = CheckpointTable::getInstance()->getCheckpoint(result, checkpointUuid, userContext, error);
    if (ret == ERROR) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }
    if (ret == INVALID_INPUT) {
        status.errorMessage = "Chekckpoint with uuid '" + checkpointUuid + "' not found";
        status.statusCode = NOT_FOUND_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // delete entry from db
    ret = CheckpointTable::getInstance()->deleteCheckpoint(checkpointUuid, userContext, error);
    if (ret != OK) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete local files
    if (Hanami::deleteFileOrDir(result.location, error) == false) {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
