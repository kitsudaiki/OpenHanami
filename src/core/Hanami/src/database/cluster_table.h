/**
 * @file        cluster_table.h
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

#ifndef CLUSTERTABLE_H
#define CLUSTERTABLE_H

#include <libKitsunemimiCommon/logger.h>
#include <database/generic_tables/hanami_sql_table.h>

namespace Kitsunemimi {
class JsonItem;
}
class ClusterTable
        : public Kitsunemimi::Hanami::HanamiSqlTable
{
public:
    ClusterTable(Kitsunemimi::Sakura::SqlDatabase* db);
    ~ClusterTable();

    bool addCluster(Kitsunemimi::JsonItem &clusterData,
                    const Kitsunemimi::Hanami::UserContext &userContext,
                    Kitsunemimi::ErrorContainer &error);
    bool getCluster(Kitsunemimi::JsonItem &result,
                    const std::string &clusterUuid,
                    const Kitsunemimi::Hanami::UserContext &userContext,
                    Kitsunemimi::ErrorContainer &error,
                    const bool showHiddenValues = false);
    bool getClusterByName(Kitsunemimi::JsonItem &result,
                          const std::string &clusterName,
                          const Kitsunemimi::Hanami::UserContext &userContext,
                          Kitsunemimi::ErrorContainer &error,
                          const bool showHiddenValues = false);
    bool getAllCluster(Kitsunemimi::TableItem &result,
                       const Kitsunemimi::Hanami::UserContext &userContext,
                       Kitsunemimi::ErrorContainer &error);
    bool deleteCluster(const std::string &clusterUuid,
                       const Kitsunemimi::Hanami::UserContext &userContext,
                       Kitsunemimi::ErrorContainer &error);
};

#endif // CLUSTERTABLE_H
