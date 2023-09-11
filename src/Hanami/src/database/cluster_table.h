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

#ifndef HANAMI_CLUSTERTABLE_H
#define HANAMI_CLUSTERTABLE_H

#include <hanami_common/logger.h>
#include <database/generic_tables/hanami_sql_table.h>

class ClusterTable
        : public HanamiSqlTable
{
public:
    static ClusterTable* getInstance()
    {
        if(instance == nullptr) {
            instance = new ClusterTable();
        }
        return instance;
    }

    ~ClusterTable();

    bool addCluster(json &clusterData,
                    const UserContext &userContext,
                    Hanami::ErrorContainer &error);
    bool getCluster(json &result,
                    const std::string &clusterUuid,
                    const UserContext &userContext,
                    Hanami::ErrorContainer &error,
                    const bool showHiddenValues = false);
    bool getClusterByName(json &result,
                          const std::string &clusterName,
                          const UserContext &userContext,
                          Hanami::ErrorContainer &error,
                          const bool showHiddenValues = false);
    bool getAllCluster(Hanami::TableItem &result,
                       const UserContext &userContext,
                       Hanami::ErrorContainer &error);
    bool deleteCluster(const std::string &clusterUuid,
                       const UserContext &userContext,
                       Hanami::ErrorContainer &error);
    bool deleteAllCluster(Hanami::ErrorContainer &error);
private:
    ClusterTable();
    static ClusterTable* instance;
};

#endif // HANAMI_CLUSTERTABLE_H
