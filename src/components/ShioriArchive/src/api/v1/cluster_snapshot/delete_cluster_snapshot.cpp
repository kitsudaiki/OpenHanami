/**
 * @file        delete_cluster_snapshot.cpp
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

#include "delete_cluster_snapshot.h"

#include <shiori_root.h>
#include <database/cluster_snapshot_table.h>

#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/methods/file_methods.h>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/defines.h>

using namespace Kitsunemimi::Hanami;

DeleteClusterSnapshot::DeleteClusterSnapshot()
    : Blossom("Delete a result-set from shiori.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the cluster-snapshot to delete.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
DeleteClusterSnapshot::runTask(BlossomIO &blossomIO,
                               const Kitsunemimi::DataMap &context,
                               BlossomStatus &status,
                               Kitsunemimi::ErrorContainer &error)
{
    const std::string dataUuid = blossomIO.input.get("uuid").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // get location from database
    Kitsunemimi::JsonItem result;
    if(ShioriRoot::clusterSnapshotTable->getClusterSnapshot(result,
                                                            dataUuid,
                                                            userContext,
                                                            error,
                                                            true) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get location from response
    const std::string location = result.get("location").getString();

    // delete entry from db
    if(ShioriRoot::clusterSnapshotTable->deleteClusterSnapshot(dataUuid,
                                                               userContext,
                                                               error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // delete local files
    if(Kitsunemimi::deleteFileOrDir(location, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
