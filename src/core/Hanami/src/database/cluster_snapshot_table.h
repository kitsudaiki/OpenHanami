/**
 * @file        cluster_snapshot_table.h
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

#ifndef HANAMI_CLUSTER_SNAPSHOT_TABLE_H
#define HANAMI_CLUSTER_SNAPSHOT_TABLE_H

#include <libKitsunemimiCommon/logger.h>
#include <database/generic_tables/hanami_sql_table.h>

namespace Kitsunemimi {
class JsonItem;
}

class ClusterSnapshotTable
        : public HanamiSqlTable
{
public:
    ClusterSnapshotTable(Kitsunemimi::Sakura::SqlDatabase* db);
    ~ClusterSnapshotTable();

    bool addClusterSnapshot(Kitsunemimi::JsonItem &data,
                            const UserContext &userContext,
                            Kitsunemimi::ErrorContainer &error);
    bool getClusterSnapshot(Kitsunemimi::JsonItem &result,
                            const std::string &snapshotUuid,
                            const UserContext &userContext,
                            Kitsunemimi::ErrorContainer &error,
                            const bool showHiddenValues);
    bool getAllClusterSnapshot(Kitsunemimi::TableItem &result,
                               const UserContext &userContext,
                               Kitsunemimi::ErrorContainer &error);
    bool deleteClusterSnapshot(const std::string &snapshotUuid,
                               const UserContext &userContext,
                               Kitsunemimi::ErrorContainer &error);
    bool setUploadFinish(const std::string &uuid,
                         const std::string &fileUuid,
                         Kitsunemimi::ErrorContainer &error);
};

#endif // HANAMI_CLUSTER_SNAPSHOT_TABLE_H
