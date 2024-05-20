/**
 * @file        cluster_table.cpp
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

#include "cluster_table.h"

#include <hanami_common/functions/string_functions.h>
#include <hanami_common/items/table_item.h>
#include <hanami_database/sql_database.h>

ClusterTable* ClusterTable::instance = nullptr;

/**
 * @brief constructor
 */
ClusterTable::ClusterTable() : HanamiSqlTable(Hanami::SqlDatabase::getInstance())
{
    m_tableName = "clusters";
}

/**
 * @brief destructor
 */
ClusterTable::~ClusterTable() {}

/**
 * @brief add a new cluster to the database
 *
 * @param userData json-item with all information of the cluster to add to database
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::addCluster(json& clusterData,
                         const Hanami::UserContext& userContext,
                         Hanami::ErrorContainer& error)
{
    if (add(clusterData, userContext, error) == false) {
        error.addMessage("Failed to add cluster-meta to database");
        return false;
    }

    return true;
}

/**
 * @brief get a cluster from the database by his name
 *
 * @param result reference for the result-output in case that a cluster with this name was found
 * @param clusterUuid uuid of the requested cluster
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::getCluster(json& result,
                         const std::string& clusterUuid,
                         const Hanami::UserContext& userContext,
                         Hanami::ErrorContainer& error,
                         const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", clusterUuid);

    // get user from db
    if (get(result, userContext, conditions, error, showHiddenValues) == false) {
        error.addMessage("Failed to get cluster-meta with UUID '" + clusterUuid
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief get a cluster from the database by his name
 *
 * @param result reference for the result-output in case that a cluster with this name was found
 * @param clusterName name of the requested cluster
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::getClusterByName(json& result,
                               const std::string& clusterName,
                               const Hanami::UserContext& userContext,
                               Hanami::ErrorContainer& error,
                               const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", clusterName);

    // get user from db
    if (get(result, userContext, conditions, error, showHiddenValues) == false) {
        error.addMessage("Failed to get cluster-meta from database by name '" + clusterName + "'");
        return false;
    }

    return true;
}

/**
 * @brief get all clusters from the database table
 *
 * @param result reference for the result-output
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::getAllCluster(Hanami::TableItem& result,
                            const Hanami::UserContext& userContext,
                            Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    if (getAll(result, userContext, conditions, error) == false) {
        error.addMessage("Failed to get all cluster-meta from database");
        return false;
    }

    return true;
}

/**
 * @brief delete a cluster from the table
 *
 * @param clusterUuid uuid of the cluster to delete
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::deleteCluster(const std::string& clusterUuid,
                            const Hanami::UserContext& userContext,
                            Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", clusterUuid);

    if (del(conditions, userContext, error) == false) {
        error.addMessage("Failed to delete cluster-meta with UUID '" + clusterUuid
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief delete all cluster from database. This currently used to avoid
 *        broken empty clusters after a restart of the backend
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ClusterTable::deleteAllCluster(Hanami::ErrorContainer& error)
{
    return deleteAllFromDb(error);
}
